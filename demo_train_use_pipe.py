import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import argparse
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from tqdm import tqdm
# PEFT 라이브러리를 통해 LoRA 모듈 적용
try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    raise ImportError("Please install peft library: pip install peft")

# latent diffusion 관련 라이브러리 (예: diffusers의 DDPMScheduler)
try:
    from diffusers import DDPMScheduler
except ImportError:
    raise ImportError("Please install diffusers library: pip install diffusers")

# CatVTONPipeline 임포트 (여러분의 프로젝트 구조에 맞게 수정)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"CatVTON")))
from model.pipeline import CatVTONPipeline
from utils import compute_vae_encodings, tensor_to_image, numpy_to_pil

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
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1,
        help="Number of inference steps to perform.",
    )
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
        pair_txt = os.path.join(self.args.data_root_path, 'train_pairs_sample.txt')
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
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 파이프라인과 모델 초기화 (여러분의 환경에 맞게 수정)
    base_ckpt = "booksforcharlie/stable-diffusion-inpainting"
    attn_ckpt = "zhengchong/CatVTON"
    attn_ckpt_version = "mix"

    pipeline = CatVTONPipeline(
        base_ckpt, 
        attn_ckpt,
        attn_ckpt_version,
        weight_dtype=torch.float32,
        device="cuda",
        skip_safety_check=True,
    )

    # fine-tuning 대상 모델 (예: UNet) 추출
    model = pipeline.unet 
    model.to(device)

    # 2. LoRA 설정 및 적용
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["to_q", "to_k", "to_v"],  
    )
    model = get_peft_model(model, lora_config)
    print("LoRA 적용 완료. 현재 학습 파라미터 수:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))


    
    # 4. 옵티마이저 및 손실 함수 정의
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()  # diffusion 학습에서는 주로 예측한 노이즈와 실제 노이즈 간의 MSE를 사용


    dataset = VITONHDTrainDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    best_loss = float("inf")
    best_epoch = -1
    model.train()
    for epoch in tqdm(range(args.num_epochs), desc="Epoch", total=args.num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            # 배치에서 person, cloth, mask 텐서를 device로 이동
            person = batch["person"].to(device)
            cloth  = batch["cloth"].to(device)
            mask   = batch["mask"].to(device)
            
            image = pipeline(
                person,
                cloth,
                mask,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
                num_inference_steps = args.num_inference_steps,
            )
            print("\n person, image :",person.shape, image.shape)
            loss = loss_fn(person, image)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1

            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, f"best_model_{best_epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"새로운 베스트 모델 저장: {checkpoint_path} (Epoch {best_epoch})")


    print("학습 완료.")


if __name__ == "__main__":
    main()