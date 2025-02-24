from io import BytesIO
from PIL import Image, ExifTags, ImageOps
from typing import Optional
import os
import sys
import torch
import re
import time
from huggingface_hub import snapshot_download, hf_hub_download
from diffusers.image_processor import VaeImageProcessor
from peft import get_peft_model, LoraConfig

# CatVTON 경로 설정
catvton_path = os.path.abspath(os.path.join(os.getcwd(), "CatVTON"))
sys.path.append(catvton_path)
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import init_weight_dtype, resize_and_crop, resize_and_padding, prepare_image, prepare_mask_image, tensor_to_image

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

def process_tryon_experiment(
    body_image_path: str,  # 로컬 사람 이미지 파일 경로
    cloth_image_path: str,  # 로컬 옷 이미지 파일 경로
    cloth_type: str,        # 필수
    lora_weight_name: Optional[str] = None  # LoRA 가중치 이름 (선택 사항)
):
    # 로컬 이미지 파일 로드
    person_image = Image.open(body_image_path).convert("RGB")
    cloth_image = Image.open(cloth_image_path).convert("RGB")

    # 파이프라인 생성
    base_ckpt_path = snapshot_download(repo_id="booksforcharlie/stable-diffusion-inpainting", local_dir="./base_ckpt")
    attn_ckpt_path = snapshot_download(repo_id="zhengchong/CatVTON", local_dir="./attn_ckpt")
    pipeline = CatVTONPipeline(
        base_ckpt=base_ckpt_path,
        attn_ckpt=attn_ckpt_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype("fp16"),
        use_tf32=True,
        device=device,
        skip_safety_check=True,
    )

    # LoRA 적용 (필요한 경우)
    if lora_weight_name and lora_weight_name != "none":
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
        pipeline.unet = get_peft_model(pipeline.unet, lora_config)
        pipeline.unet.load_state_dict(torch.load(lora_weights_path, map_location=device), strict=False)
        print(f"\n*******Loaded LoRA weights into pipeline from {lora_weights_path}")

    # 추론 
    with torch.no_grad():
        person_image = ImageOps.exif_transpose(person_image)
        cloth_image = ImageOps.exif_transpose(cloth_image)
        person_image = resize_and_crop(person_image, (768, 1024))
        cloth_image = resize_and_padding(cloth_image, (768, 1024))

        mask = automasker(person_image, mask_type=cloth_type)['mask']

        generator = torch.Generator(device=device).manual_seed(555)
        result = pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=50,
            guidance_scale=2.5,
            height=1024,
            width=768,
            generator=generator,
        )[0]

    # 결과 이미지를 로컬에 저장
    output_path = os.path.join("tryon-images", f"result_{int(time.time())}.png")
    result.save(output_path, format="PNG")
    return output_path

if __name__ == "__main__":
    #C:\Users\coldbrew\VTON-project\tryon-images\하의2.png
    body_image_path = "C:/Users/coldbrew/VTON-project/tryon-images/IMG_7480.JPG"  # 실제 사람 이미지 경로
    cloth_image_path = "C:/Users/coldbrew/VTON-project/tryon-images/lv1.jpg"  # 실제 옷 이미지 경로로 변경
    cloth_type = "upper"  # 예: "upper_body", "lower_body" 등
    lora_weight_name = "best_lpips_lora_r32_ep20_20250223_001931.pt"  # "none" 또는 특정 LoRA 가중치 파일 이름
    # lora_weight_name = ""
    result_path = process_tryon_experiment(body_image_path, cloth_image_path, cloth_type, lora_weight_name)
    print(f"Result saved at: {result_path}")