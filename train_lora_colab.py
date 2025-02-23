import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from peft import get_peft_model, LoraConfig, TaskType
from peft import get_peft_model_state_dict

from diffusers.image_processor import VaeImageProcessor

import numpy as np
import cv2 
from PIL import Image
from tqdm import tqdm

#metric
from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from torchvision.transforms.functional import normalize
import lpips

# 윈도우에서 wandb 사용 시 필요한 코드
import resource
if not hasattr(resource, "getpagesize"):
    resource.getpagesize = lambda: 4096
import wandb

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "CatVTON")))
from model.pipeline_train import CatVTONPipeline_Train
from utils import compute_vae_encodings, tensor_to_image, numpy_to_pil
from accelerate import Accelerator

# 모델 배포 과정
from huggingface_hub import HfApi
from datetime import datetime
api = HfApi()
# dagshub + mlflow로 기록
import dagshub
import mlflow



def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Latent Diffusion based CatVTON")
    parser.add_argument("--data_root_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save checkpoints.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank parameter.")
    parser.add_argument("--seed", type=int, default=555, help="Random seed for reproducibility.")
    parser.add_argument("--eval_pair",action="store_true", default=True, help="Evaluate on paired images.")
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--width", type=int, default=768, help="Image width.")
    parser.add_argument("--use_tf32", default=False, help="Use TF32 precision for training.")
    parser.add_argument("--attn_ckpt_version", type=str, default="mix", help="Version of the attention checkpoint.")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="Guidance scale for the diffusion model.")
    parser.add_argument("--use_fp16", default=False, help="Use FP16 precision for training.")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients before update.")
    parser.add_argument("--use_maked_loss", action="store_true", default=False, help="Use masked loss for training.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for validation")
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
    
class Custom_VITONHDTrainDataset(TrainDataset):
    def load_data(self):
        pair_txt = os.path.join(self.args.data_root_path, 'train_unpair.txt')
        assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            lines = f.readlines()
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
                'cloth_l': os.path.join(self.args.data_root_path, 'cloth', 'lower_img', cloth_img),
                'cloth_u': os.path.join(self.args.data_root_path, 'cloth', 'upper_img', cloth_img),
                'mask_l': os.path.join(self.args.data_root_path, 'image_mask_L', person_img),
                'mask_u': os.path.join(self.args.data_root_path, 'image_mask_U', person_img)
            })
        return data

    def __getitem__(self, idx):
        data = self.data[idx]
        person = Image.open(data['person'])

        # 상의(cloth)와 하의(cloth_l) 중 랜덤 선택
        if random.random() < 0.5:
            cloth = Image.open(data['cloth_u'])  # 상의
            mask = Image.open(data['mask_u'])    # 상의 마스크
        else:
            cloth = Image.open(data['cloth_l'])  # 하의
            mask = Image.open(data['mask_l'])    # 하의 마스크

        # Horizontal flip 적용
        if random.random() < 0.5:
            person = person.transpose(Image.FLIP_LEFT_RIGHT)
            cloth = cloth.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {
            'index': idx,
            'person_name': data['person_name'],
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            'mask': self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0]
        }

class Custom_VITONHDTestDataset(TrainDataset):
    def load_data(self):
        pair_txt = os.path.join(self.args.data_root_path, 'test_unpair.txt')
        assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            lines = f.readlines()
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
                'cloth_l': os.path.join(self.args.data_root_path, 'cloth', 'lower_img', cloth_img),
                'cloth_u': os.path.join(self.args.data_root_path, 'cloth', 'upper_img', cloth_img),
                'mask_l': os.path.join(self.args.data_root_path, 'image_mask_L', person_img),
                'mask_u': os.path.join(self.args.data_root_path, 'image_mask_U', person_img)
            })
        return data

    def __getitem__(self, idx):
        data = self.data[idx]
        person = Image.open(data['person'])

        # 상의(cloth)와 하의(cloth_l) 중 랜덤 선택
        if random.random() < 0.5:
            cloth = Image.open(data['cloth_u'])  # 상의
            mask = Image.open(data['mask_u'])    # 상의 마스크
        else:
            cloth = Image.open(data['cloth_l'])  # 하의
            mask = Image.open(data['mask_l'])    # 하의 마스크

        # Horizontal flip 적용
        if random.random() < 0.5:
            person = person.transpose(Image.FLIP_LEFT_RIGHT)
            cloth = cloth.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {
            'index': idx,
            'person_name': data['person_name'],
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            'mask': self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0]
        }

class VITONHDTestDataset(TrainDataset):
    def load_data(self):
        pair_txt = os.path.join(self.args.data_root_path, 'test_pairs_unpaired.txt')
        assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            lines = f.readlines()
        self.args.data_root_path = os.path.join(self.args.data_root_path, "test")
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
                'mask': os.path.join(self.args.data_root_path, 'agnostic-mask', person_img.replace('.jpg', '_mask.png')),
            })
        return data

# class VITONHDTrainDataset(TrainDataset):
#     def load_data(self):
#         pair_txt = os.path.join(self.args.data_root_path, 'train_pairs.txt')
#         assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
#         with open(pair_txt, 'r') as f:
#             lines = f.readlines()
#         self.args.data_root_path = os.path.join(self.args.data_root_path, "train")
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
#                 'mask': os.path.join(self.args.data_root_path, 'agnostic-v3.2', person_img),
#             })
#         return data

def to_pil_image(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # normalize to [0, 1]
        if float32:
            img = img.astype(np.float32) #/ 255.
        img = torch.from_numpy(img.transpose(2, 0, 1))        
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.use_fp16:
        precision = "fp16"
    else:
        precision = "fp32"
        
    dagshub.init(repo_owner='ColdTbrew', repo_name='VTON-project', mlflow=True)
    wandb.init(project="VTON-project", config=vars(args))
    # MLflow 실험 시작 및 Auto-logging 활성화
    mlflow.set_experiment("CatVTON_LoRA_Training")
    run_name = os.path.join(args.output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))
        mlflow.pytorch.autolog()
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        accelerator = Accelerator(mixed_precision=precision)
        device = accelerator.device

        # 1. 파이프라인과 모델 초기화 (여러분의 환경에 맞게 수정)
        base_ckpt = "booksforcharlie/stable-diffusion-inpainting"
        attn_ckpt = "zhengchong/CatVTON"
        attn_ckpt_version = "mix"
        lpips_model_name = 'vgg'

        pipeline = CatVTONPipeline_Train(
            base_ckpt, 
            attn_ckpt,
            attn_ckpt_version,
            device=device,
            skip_safety_check=True,
        )
        pipeline.vae.eval()

        # VAE 모델의 파라미터를 고정
        for param in pipeline.vae.parameters():
            param.requires_grad = False
        # UNet 모델의 어텐션 제외 파라미터를 고정
        pipeline.unet.train()
        for name, param in pipeline.unet.named_parameters():
            if "attention" not in name:
                param.requires_grad = False

        model = pipeline.unet 
        model = model.to(device)

        # 2. LoRA 설정 및 적용
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["to_q", "to_k", "to_v"],  
        )
        model = get_peft_model(model, lora_config)
        model = model.to(device)

        print("LoRA 적용 완료. 현재 학습 파라미터 수:",
            sum(p.numel() for p in model.parameters() if p.requires_grad))
        pipeline.unet = model

        # 4. 옵티마이저 및 손실 함수 정의
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        # dataset = VITONHDTrainDataset(args)
        dataset = Custom_VITONHDTrainDataset(args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        val_dataset = Custom_VITONHDTestDataset(args)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Accelerator로 모델, 옵티마이저, 데이터로더 준비
        model, optimizer, dataloader, val_dataloader = accelerator.prepare(model, optimizer, dataloader, val_dataloader)

        generator = torch.Generator(device='cuda').manual_seed(args.seed)
        best_loss = float("inf")
        best_PSNR = float("-inf")  # 클수록 좋음
        best_SSIM = float("-inf")  # 클수록 좋음
        best_LPIPS = float("inf")  # 작을수록 좋음

        best_epoch = -1
        global_step = 0
        model.train()
        
        # lpips
        if lpips_model_name == 'vgg':
            lpips_vgg = lpips.LPIPS(net='vgg').to(device)
        if lpips_model_name == 'alex':
            lpips_alex = lpips.LPIPS(net='alex').to(device)

        device = accelerator.device
        for epoch in tqdm(range(args.num_epochs), desc="Epoch", total=args.num_epochs):
            total_loss = 0.0
            optimizer.zero_grad()  # 옵티마이저 초기화
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(model):  # Gradient accumulation 시작
                    person = batch["person"]
                    cloth = batch["cloth"]
                    mask = batch["mask"]

                    with accelerator.autocast():
                        noise, noise_pred = pipeline(
                            person,
                            cloth,
                            mask,
                            guidance_scale=args.guidance_scale,
                            height=args.height,
                            width=args.width,
                            generator=generator,
                            is_train=True,  # 학습 모드
                        )
                        loss = loss_fn(noise, noise_pred)
                    
                    # 그라디언트 계산
                    accelerator.backward(loss / args.accumulation_steps)  # 손실을 나눠서 누적
                    
                    # accumulation_steps마다 업데이트
                    if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(dataloader):
                        optimizer.step()
                        optimizer.zero_grad()

                    total_loss += loss.item()
                    global_step += 1
                    wandb.log({"train_loss": loss.item(), "global_step": global_step})
            avg_loss = total_loss / len(dataloader)

            # print and log
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}")
            wandb.log({"avg_loss": avg_loss, "epoch": epoch+1})
            # Auto-logging이 손실을 이미 로깅하면 이 줄 제거 가능
            # mlflow.log_metrics({"avg_loss": avg_loss}, step=epoch)

            # 5. 에폭마다 모델 각각 저장
            os.makedirs(args.output_dir, exist_ok=True)
            if (epoch+1) % 5 == 0:
                # validation
                model.eval()
                val_psnr, val_ssim, val_lpips = [], [], []
                with torch.no_grad():
                    generator = torch.Generator(device='cuda').manual_seed(args.seed)
                    with accelerator.autocast():
                        for batch in tqdm(val_dataloader, desc="Validation", total=len(val_dataloader)):
                            person = batch["person"]
                            cloth  = batch["cloth"]
                            mask   = batch["mask"]
                            results = pipeline(
                                person,
                                cloth,
                                mask,
                                num_inference_steps=args.num_inference_steps,
                                guidance_scale=args.guidance_scale,
                                height=args.height,
                                width=args.width,
                                generator=generator,
                                is_train=False,  # 추론 모드
                            )

                            gt_person = to_pil_image(person)
                            for i, (gt_img, pred_img) in enumerate(zip(gt_person, results)):
                                gt_np = np.array(gt_img)
                                pred_np = np.array(pred_img)
                                psnr_val = calculate_psnr(gt_np, pred_np, crop_border=0)
                                ssim_val = calculate_ssim(gt_np, pred_np, crop_border=0)
                                val_psnr.append(psnr_val)
                                val_ssim.append(ssim_val)

                                gt_tensor, pred_tensor = img2tensor([gt_np, pred_np], bgr2rgb=True, float32=True)
                                normalize(gt_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
                                normalize(pred_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
                                lpips_val = lpips_vgg(gt_tensor.unsqueeze(0).to(device), pred_tensor.unsqueeze(0).to(device)).cpu().item()
                                val_lpips.append(lpips_val)

                                output_dir = os.path.join(args.output_dir, f"val_epoch_{epoch+1}")
                                os.makedirs(output_dir, exist_ok=True)
                                gt_img.save(os.path.join(output_dir, f"{batch['person_name'][i]}_gt.jpg"))
                                pred_img.save(os.path.join(output_dir, f"{batch['person_name'][i]}_pred.jpg"))

                avg_psnr = np.mean(val_psnr)
                avg_ssim = np.mean(val_ssim)
                avg_lpips = np.mean(val_lpips)
                print(f"Validation - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
                wandb.log({
                    "val_psnr": avg_psnr, 
                    "val_ssim": avg_ssim, 
                    "val_lpips": avg_lpips, 
                    })
                mlflow.log_metrics({
                    "val_psnr": avg_psnr,
                    "val_ssim": avg_ssim,
                    "val_lpips": avg_lpips
                }, step=global_step)

                # 모델 저장 (LPIPS 기준으로 최적 모델 저장)
                if avg_lpips < best_LPIPS:
                    best_LPIPS = avg_lpips
                    best_epoch = epoch + 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkpoint_filename = f"best_lpips_lora_r{args.lora_rank}_ep{best_epoch}_{timestamp}.pt"
                    checkpoint_path = os.path.join(args.output_dir, "best_checkpoint", checkpoint_filename)
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    lora_state_dict = get_peft_model_state_dict(model)
                    torch.save(lora_state_dict, checkpoint_path)
                    api.upload_file(
                        path_or_fileobj=checkpoint_path,
                        path_in_repo=checkpoint_filename,
                        repo_id="Coldbrew9/wheel-CatVTON",
                        repo_type="model",
                    )
                    print(f"best lpips lora 모델 저장: {checkpoint_path} (Epoch {best_epoch})")
                    wandb.log({"best_lpips": best_LPIPS, "best_epoch": best_epoch, "epoch": epoch+1})
                    wandb.run.summary["best_lpips"] = best_LPIPS
                    wandb.run.summary["best_epoch"] = best_epoch
                    # 태그 업데이트 (mlflow)
                    mlflow.set_tag("best_lpips", best_LPIPS)
                    mlflow.set_tag("best_epoch", best_epoch)
                model.train()

            # loss가 감소하면 모델 저장
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                checkpoint_filename = f"best_loss_lora_r{args.lora_rank}_ep{best_epoch}_{timestamp}.pt"
                checkpoint_path = os.path.join(args.output_dir, checkpoint_filename)
                os.makedirs(args.output_dir, exist_ok=True)
                
                # 모델 저장
                lora_state_dict = get_peft_model_state_dict(model)
                torch.save(lora_state_dict, checkpoint_path)
                print(f"loss best lora 모델 저장: {checkpoint_path} (Epoch {best_epoch})")

                # Hugging Face에 업로드
                try:
                    api.upload_file(
                        path_or_fileobj=checkpoint_path,
                        path_in_repo=checkpoint_filename,
                        repo_id="Coldbrew9/wheel-CatVTON",
                        repo_type="model",
                    )
                    print(f"모델 업로드 완료: {checkpoint_filename} -> Coldbrew9/wheel-CatVTON")
                except Exception as e:
                    print(f"업로드 실패: {e}")
                
                wandb.log({"best_loss": best_loss, "best_epoch": best_epoch, "epoch": epoch+1})
                # 태그 업데이트 (mlflow)
                mlflow.set_tag("best_epoch", best_epoch)
                mlflow.set_tag("best_loss", best_loss)

                # wandb에 베스트 모델 업데이트 기록
                wandb.run.summary["best_loss"] = best_loss
                wandb.run.summary["best_epoch"] = best_epoch

    print("학습 완료.")
    wandb.finish()
    mlflow.end_run()


if __name__ == "__main__":
    main()