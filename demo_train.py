import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm import tqdm
import resource
if not hasattr(resource, "getpagesize"):
    resource.getpagesize = lambda: 4096
import wandb
try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    raise ImportError("Please install peft library: pip install peft")

try:
    from diffusers import DDPMScheduler
except ImportError:
    raise ImportError("Please install diffusers library: pip install diffusers")

# CatVTONPipeline 임포트 (여러분의 프로젝트 구조에 맞게 수정)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"CatVTON")))
from model.pipeline import CatVTONPipeline
from model.utils import get_trainable_module, init_adapter
from utils import compute_vae_encodings
from model.attn_processor import SkipAttnProcessor
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Latent Diffusion based CatVTON")
    parser.add_argument("--data_root_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save checkpoints.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank parameter.")
    parser.add_argument("--seed", type=int, default=555, help="Random seed for reproducibility.")
    parser.add_argument("--eval_pair", default=False, help="Evaluate on paired images.")
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--width", type=int, default=768, help="Image width.")
    # latent diffusion 관련 파라미터
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of diffusion steps for training.")
    parser.add_argument("--use_tf32", default=False, help="Use TF32 precision for training.")
    parser.add_argument("--attn_ckpt_version", type=str, default="mix", help="Version of the attention checkpoint.")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="Guidance scale for the diffusion model.")
    args = parser.parse_args()
    return args


class TrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8, 
            do_normalize=False, 
            do_binarize=True, 
            do_convert_grayscale=True
        )
        self.data = self.load_data()

    def load_data(self):
        return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        person, cloth, mask = [Image.open(data[key]) for key in ['person', 'cloth', 'mask']]
        return {
            'index': idx,
            'person_name': data['person_name'],
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            'mask': self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0]
        }


# class VITONHDTestDataset(TrainDataset):
#     def load_data(self):
#         pair_txt = os.path.join(self.args.data_root_path, 'test_pairs_unpaired.txt')
#         assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
#         with open(pair_txt, 'r') as f:
#             lines = f.readlines()
#         self.args.data_root_path = os.path.join(self.args.data_root_path, "test")
#         output_dir = os.path.join(
#             self.args.output_dir, 
#             "vitonhd", 
#             'unpaired' if not self.args.eval_pair else 'paired'
#         )
#         data = []
#         for line in lines:
#             person_img, cloth_img = line.strip().split(" ")
#             if os.path.exists(os.path.join(output_dir, person_img)):
#                 continue
#             if self.args.eval_pair:
#                 cloth_img = person_img
#             data.append({
#                 'person_name': person_img,
#                 'person': os.path.join(self.args.data_root_path, 'image', person_img),
#                 'cloth': os.path.join(self.args.data_root_path, 'cloth', cloth_img),
#                 'mask': os.path.join(self.args.data_root_path, 'agnostic-mask', person_img.replace('.jpg', '_mask.png')),
#             })
#         return data

class VITONHDTrainDataset(TrainDataset):
    def load_data(self):
        pair_txt = os.path.join(self.args.data_root_path, 'train_pairs.txt')
        assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            lines = f.readlines()
        self.args.data_root_path = os.path.join(self.args.data_root_path, "train")
        output_dir = os.path.join(
            self.args.output_dir, 
            "vitonhd", 
            'unpaired' if not self.args.eval_pair else 'paired'
        )
        data = []
        for line in lines:
            person_img, cloth_img = line.strip().split(" ")
            if os.path.exists(os.path.join(output_dir, person_img)):
                continue
            if self.args.eval_pair:
                cloth_img = person_img
            data.append({
                'person_name': person_img,
                'person': os.path.join(self.args.data_root_path, 'image', person_img),
                'cloth': os.path.join(self.args.data_root_path, 'cloth', cloth_img),
                'mask': os.path.join(self.args.data_root_path, 'agnostic-v3.2', person_img),
            })
        return data


def main():
    args = parse_args()
    # wandb 초기화 원하지 않으면 주석 처리
    wandb.init(project="VTON-project", config=vars(args))
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_ckpt = "booksforcharlie/stable-diffusion-inpainting"
    attn_ckpt = "zhengchong/CatVTON"
    attn_ckpt_version = "mix"

    pipeline = CatVTONPipeline(
        base_ckpt, 
        attn_ckpt,
        attn_ckpt_version,
        weight_dtype=torch.float32,
        device="cuda",
        skip_safety_check=True
    )
    if args.use_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    # fine-tuning 대상 모델 (예: UNet) 추출
    model = pipeline.unet 
    model.to(device)
    # 필요해 보임
    init_adapter(pipeline.unet, cross_attn_cls=SkipAttnProcessor)  # Skip Cross-Attention
    pipeline.attn_modules = get_trainable_module(pipeline.unet, "attention")
    pipeline.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)

    # 2. LoRA 설정 및 적용
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=6,
        lora_dropout=0.1,
        target_modules=["to_q", "to_k", "to_v"],  
    )
    model = get_peft_model(model, lora_config)
    print("LoRA 적용 완료. 현재 학습 파라미터 수:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 3. 노이즈 스케줄러 (DDPM) 초기화
    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    # self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")

    # 4. 옵티마이저 및 손실 함수 정의
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()  # diffusion 학습에서는 주로 예측한 노이즈와 실제 노이즈 간의 MSE를 사용

    dataset = VITONHDTrainDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    best_loss = float("inf")
    best_epoch = -1
    global_step = 0
    concat_dim = -2
    model.train()
    for epoch in tqdm(range(args.num_epochs), desc="Epoch", total=args.num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()

            # 텐서들을 device로 이동
            person = batch["person"].to(device)
            condition_image = batch["cloth"].to(device)
            mask = batch["mask"].to(device)
            
            # Inpainting 조건 구성: 마스크 영역 (mask < 0.5)
            masked_image = person * (mask < 0.5).float()

            # VAE 인코딩 (출력 shape: [B, 4, H', W'])
            masked_latent = compute_vae_encodings(masked_image, pipeline.vae)
            condition_latent = compute_vae_encodings(condition_image, pipeline.vae)
            # mask를 latent 크기로 resize (출력 shape: [B, 1, H', W'])
            mask_latent = F.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
            del person, mask, condition_image

            # latent 결합 (concat_dim = -2, 즉 높이 축에서 결합)
            # masked_latent와 condition_latent를 결합하면 [B, 4, 2H', W']
            masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
            # mask_latent와 같은 shape의 zeros를 결합하여 [B, 1, 2H', W']
            mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)

            # 노이즈 추가 (diffusion training)
            non_inpainting_latent = torch.randn_like(masked_latent)
            noise = torch.randn_like(non_inpainting_latent)
            batch_size = non_inpainting_latent.shape[0]
            timesteps = torch.randint(0, args.num_train_timesteps, (batch_size,), device=device).long()
            noisy_latents = scheduler.add_noise(non_inpainting_latent, noise, timesteps)
            non_inpainting_scaled = scheduler.scale_model_input(noisy_latents, timesteps)
            # non_inpainting_scaled의 높이를 masked_latent_concat과 맞춤
            non_inpainting_scaled = F.interpolate(
                non_inpainting_scaled,
                size=(masked_latent_concat.shape[-2], masked_latent_concat.shape[-1]),
                mode="nearest"
            )
            print("masked_image", masked_image.shape)
            print("masked_latent", masked_latent.shape)
            print("condition_latent", condition_latent.shape)
            print("mask_latent", mask_latent.shape)
            print("masked_latent_concat", masked_latent_concat.shape)
            print("mask_latent_concat", mask_latent_concat.shape)

            # 최종 모델 입력 구성: 
            # non_inpainting_scaled (4채널, [B,4,2H',W']),
            # mask_latent_concat (1채널, [B,1,2H',W']),
            # masked_latent_concat (4채널, [B,4,2H',W'])
            # → 총 9채널
            model_input = torch.cat([non_inpainting_scaled, mask_latent_concat, masked_latent_concat], dim=concat_dim)
            print("model_input", model_input.shape)

            # 모델 예측 (UNet은 timestep과 함께 입력받음)
            predicted_noise = model(model_input, encoder_hidden_states=None, timestep=timesteps)[0]
            print("model_output", predicted_noise.shape)
            # 손실 계산 및 역전파
            loss = loss_fn(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1
            wandb.log({"train_loss": loss.item(), "global_step": global_step})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}")

        # 베스트 성능 체크: 현재 에폭의 손실이 더 낮으면 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1  # 에폭 번호는 1부터 시작

            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, f"best_model_{best_epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"새로운 베스트 모델 저장: {checkpoint_path} (Epoch {best_epoch})")
            
            # wandb에 베스트 모델 업데이트 기록
            wandb.run.summary["best_loss"] = best_loss
            wandb.run.summary["best_epoch"] = best_epoch

    print("학습 완료.")
    wandb.finish()

if __name__ == "__main__":
    main()