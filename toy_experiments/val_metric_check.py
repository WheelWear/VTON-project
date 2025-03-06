from io import BytesIO
from PIL import Image, ImageOps
from typing import Optional
import os
import sys
import torch
import re
import time
from huggingface_hub import snapshot_download, hf_hub_download
from diffusers.image_processor import VaeImageProcessor
from peft import get_peft_model, LoraConfig
from PIL import Image, ImageFilter
import numpy as np
# CatVTON 경로 설정
catvton_path = os.path.abspath(os.path.join(os.getcwd(), "CatVTON"))
sys.path.append(catvton_path)
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import init_weight_dtype, resize_and_crop, resize_and_padding, prepare_image, prepare_mask_image, tensor_to_image
import random
from torch.utils.data import Dataset, DataLoader
import argparse
import lpips
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from torchvision import transforms
#import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Latent Diffusion based CatVTON")
    parser.add_argument("--data_root_path", type=str, default="./dataset", help="Path to the training dataset.")
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

# LPIPS Loss 초기화
lpips_loss = lpips.LPIPS(net='vgg').to("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_metrics(gt_img, pred_img, device):
    """PSNR, SSIM, LPIPS 계산"""
    transform = transforms.ToTensor()
    gt_tensor = transform(gt_img).unsqueeze(0).to(device)
    pred_tensor = transform(pred_img).unsqueeze(0).to(device)

    lpips_value = lpips_loss(gt_tensor, pred_tensor).item()
    
    gt_np = np.array(gt_img)
    pred_np = np.array(pred_img)
    psnr_value = calculate_psnr(gt_np, pred_np, data_range=255)
    ssim_value = calculate_ssim(gt_np, pred_np, multichannel=True, data_range=255)
    
    return psnr_value, ssim_value, lpips_value


# CatVTON 파이프라인 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA 가중치 경로 설정
lora_ckpt_base = "lora_weights"
os.makedirs(lora_ckpt_base, exist_ok=True)
repo_id = "Coldbrew9/wheel-CatVTON"

mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
repo_path = snapshot_download(repo_id="zhengchong/CatVTON")
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=device
)

# 로컬 저장 폴더
os.makedirs("tryon-images", exist_ok=True)

def repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 50
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    mask_np = np.expand_dims(mask_np, axis=-1).repeat(3, axis=-1)
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result

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

def process_tryon_experiment(
    cloth_type: str,       # 필수
    lora_weight_name: Optional[str] = None,  # LoRA 가중치 이름 (선택 사항)
    use_repaint = True
):
    args = parse_args()
    val_dataset = VITONHDTestDataset(args)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    #wandb.init(project="final-VTON-project", config=vars(args)) 
    # 기본 파이프라인 생성 (LoRA 적용 X)
    base_ckpt_path = snapshot_download(repo_id="booksforcharlie/stable-diffusion-inpainting", local_dir="./base_ckpt")
    attn_ckpt_path = snapshot_download(repo_id="zhengchong/CatVTON", local_dir="./attn_ckpt")
    generator = torch.Generator(device=device).manual_seed(555)

    # pipeline_no_lora = CatVTONPipeline(
    #     base_ckpt=base_ckpt_path,
    #     attn_ckpt=attn_ckpt_path,
    #     attn_ckpt_version="mix",
    #     weight_dtype=init_weight_dtype("fp16"),
    #     use_tf32=True,
    #     device=device,
    #     skip_safety_check=True,
    # )
    # # LoRA를 적용하지 않은 추론
    # with torch.no_grad():
    #     generator = torch.Generator(device=device).manual_seed(555)
    #     result_no_lora = pipeline_no_lora(
    #         image=person,
    #         condition_image=cloth,
    #         mask=mask,
    #         num_inference_steps=50,
    #         guidance_scale=2.5,
    #         height=1024,
    #         width=768,
    #         generator=generator,
    #     )[0]
    # del pipeline_no_lora

    # if use_repaint:
    #     result_no_lora = repaint(person, mask, result_no_lora)

    # LoRA 가중치 로드
    lora_weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=lora_weight_name,
        local_dir=lora_ckpt_base,
        repo_type="model"
    )
    match = re.search(r"lora_r(\d+)", lora_weight_name)
    lora_rank = int(match.group(1)) if match else 4
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.1,
        target_modules=["to_q", "to_k", "to_v"],
    )
    pipeline_with_lora = CatVTONPipeline(
        base_ckpt=base_ckpt_path,
        attn_ckpt=attn_ckpt_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype("fp16"),
        use_tf32=True,
        device=device,
        skip_safety_check=True,
    )
    pipeline_with_lora.unet = get_peft_model(pipeline_with_lora.unet, lora_config)
    pipeline_with_lora.unet.load_state_dict(torch.load(lora_weights_path, map_location=device), strict=False)
    print(f"\n*******Loaded LoRA weights into pipeline from {lora_weights_path}")
    
    psnr_list, ssim_list, lpips_list = [], [], []
            
    for batch in val_dataloader:
        person = batch["person"].to(device)
        cloth = batch["cloth"].to(device)
        mask = batch["mask"].to(device)
        
        with torch.no_grad():
            # result = pipeline_with_lora(
            #     image=person,
            #     condition_image=cloth,
            #     mask=mask,
            #     num_inference_steps=args.num_inference_steps,
            #     guidance_scale=args.guidance_scale,
            #     height=args.height,
            #     width=args.width
            # )[0]

            result_with_lora = pipeline_with_lora(
            image=person,
            condition_image=cloth,
            mask=mask,
            num_inference_steps=50,
            guidance_scale=2.5,
            height=1024,
            width=768,
            generator=generator,
            )[0]
    
            if use_repaint:
                result_with_lora = repaint(person, mask, result_with_lora)
        
        # GT 이미지 로드
        gt_path = os.path.join(args.data_root_path, "image", batch["person_name"][0])
        gt_img = Image.open(gt_path).convert("RGB")
        
        # PSNR, SSIM, LPIPS 계산
        psnr, ssim, lpips_val = evaluate_metrics(gt_img, result_with_lora, device)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_val)
    
    # 모델별 평균 점수 저장
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)
    print(f"Validation - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
    #wandb.log({"val_psnr": avg_psnr, "val_ssim": avg_ssim, "val_lpips": avg_lpips})


if __name__ == "__main__":
    #body_image_path = "C:/Users/coldbrew/VTON-project/final_test/person/people_1.jpg"
    #cloth_image_path ="C:/Users/coldbrew/VTON-project/final_test/cloth_u/hr_upper_09.jpg"
    cloth_type = "upper"
    lora_weight_name = "Wheel-wear_lora_r16_ep10.pt"
    use_repaint = False
    # lora_weight_name = "best_lpips_lora_r32_lr1e-05_ep35_20250224_155113.pt"
    result_path = process_tryon_experiment(cloth_type, lora_weight_name, use_repaint)
    print(f"Result saved at: {result_path}")
