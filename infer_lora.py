import os
import sys
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from diffusers import DDPMScheduler
sys.path.append(os.path.abspath(os.path.join(os.path.getcwd(), "CatVTON")))
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict
from model.cloth_masker import AutoMasker
from CatVTON.model.pipeline_train import CatVTONPipeline_Train
from CatVTON.utils import compute_vae_encodings, tensor_to_image, numpy_to_pil, prepare_image, prepare_mask_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# CatVTON 프로젝트 경로 추가
catvton_path = os.path.abspath(os.path.join(os.path.getcwd(), "CatVTON"))
if os.path.exists(catvton_path):
    sys.path.append(catvton_path)
else:
    print(f"CatVTON directory not found at {catvton_path}. Please adjust the path.")
    catvton_path = "/content/CatVTON"  # Colab에서 예시 경로
    if os.path.exists(catvton_path):
        sys.path.append(catvton_path)
    else:
        raise FileNotFoundError("CatVTON directory not found. Please specify the correct path.")

# 데이터셋 및 출력 경로 설정
data_root_path = "/content/dataset"  # Google Colab 예시, 로컬 경로로 변경 가능
output_dir = "infer"  # 결과 및 LoRA 가중치 저장 디렉토리
os.makedirs(output_dir, exist_ok=True)

# GPU 사용 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CatVTON 모델 다운로드
repo_path = snapshot_download(repo_id="zhengchong/CatVTON")
print(f"Model downloaded to: {repo_path}")

# AutoMasker 초기화
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=device,
)

# 파이프라인 초기화
pipeline = CatVTONPipeline_Train(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt="zhengchong/CatVTON",
    attn_ckpt_version="mix",
    device=device,
    skip_safety_check=True,
)

# VAE 모델의 파라미터를 고정
for param in pipeline.vae.parameters():
    param.requires_grad = False
pipeline.vae.eval()

# UNet 모델 로드 및 LoRA 적용
pipeline.unet.train()
for name, param in pipeline.unet.named_parameters():
    if "attention" not in name:
        param.requires_grad = False

# LoRA 설정
lora_rank = 4  # 이전에 사용된 LoRA rank
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_rank * 2,
    lora_dropout=0.1,
    target_modules=["to_q", "to_k", "to_v"],
)
model = get_peft_model(pipeline.unet, lora_config)
model = model.to(device)
pipeline.unet = model

# 학습된 LoRA 가중치 로드 (예: best_loss_lora_model_10.pt, 마지막 에폭 기준)
lora_weights_path = os.path.join(output_dir, "best_loss_lora_model_10.pt")  # 실제 파일명으로 수정
if os.path.exists(lora_weights_path):
    lora_state_dict = torch.load(lora_weights_path)
    model.load_state_dict(lora_state_dict, strict=False)
    print(f"Loaded LoRA weights from {lora_weights_path}")
else:
    raise FileNotFoundError(f"LoRA weights not found at {lora_weights_path}. Please provide the correct path.")

# 데이터 준비
person_path = os.path.join(data_root_path, "images", "person.jpg")  # 제공된 이미지
cloth_path = os.path.join(data_root_path, "cloth", "upper_img", "cloth.jpg")  # 예시 의류 이미지, 경로 수정 필요
cloth_type = "upper"  # 상의 또는 "lower"로 변경 가능

assert os.path.exists(person_path), f"Person image not found at {person_path}"
assert os.path.exists(cloth_path), f"Cloth image not found at {cloth_path}"

# 이미지 로드 및 전처리
person_img = Image.open(person_path).convert("RGB")
cloth_img = Image.open(cloth_path).convert("RGB")

# 이미지 크기 조정 (768x1024)
def resize_image(img, target_size=(1024, 768)):  # (height, width)
    return img.resize(target_size, Image.Resampling.LANCZOS)

person_img = resize_image(person_img, (1024, 768))
cloth_img = resize_image(cloth_img, (1024, 768))

# 무관성 마스크 생성
mask = automasker(person_path, cloth_type=cloth_type)['mask']
mask = resize_image(mask, (1024, 768))  # 마스크도 동일한 크기로 조정

# 텐서로 변환
person_tensor = prepare_image(person_img).to(device)
cloth_tensor = prepare_image(cloth_img).to(device)
mask_tensor = prepare_mask_image(mask).to(device)

# 추론 설정
num_inference_steps = 50
guidance_scale = 2.5
generator = torch.Generator(device=device).manual_seed(555)  # 고정 시드

# LoRA 적용된 파이프라인으로 추론
pipeline.unet.eval()  # 추론 모드로 전환
with torch.no_grad():
    result = pipeline(
        image=person_tensor,
        condition_image=cloth_tensor,
        mask=mask_tensor,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=1024,  # 지정된 높이
        width=768,    # 지정된 너비
        generator=generator,
        is_train=False,  # 추론 모드
    )

# 결과 이미지 변환
result_img = tensor_to_image(result[0])  # 첫 번째 배치 결과

# 이미지 저장
result_save_path = os.path.join(output_dir, "result_image.png")
result_img.save(result_save_path)
print(f"Result image saved to {result_save_path}")