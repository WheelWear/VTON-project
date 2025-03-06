from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from io import BytesIO
from PIL import Image, ExifTags, ImageOps
from typing import Optional, Dict
import os
import sys
import torch
import requests
import re
import time
import logging
from huggingface_hub import snapshot_download, hf_hub_download
from diffusers.image_processor import VaeImageProcessor
from peft import get_peft_model, LoraConfig
from google.cloud import storage
from google.oauth2 import service_account

# 디렉토리 및 로깅 설정
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app_experimental.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CatVTON 경로 설정
catvton_path = os.path.abspath(os.path.join(os.getcwd(), "CatVTON"))
sys.path.append(catvton_path)
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import init_weight_dtype, resize_and_crop, resize_and_padding, prepare_image, prepare_mask_image, tensor_to_image

# GCS 설정
credentials = service_account.Credentials.from_service_account_file('./.env/web-project-438308-a8f3849fdf23.json')
client = storage.Client(credentials=credentials, project='web-project-438308')
BUCKET_NAME = "wheelwear-bucket"

app = FastAPI()
count = 0
repo_path = snapshot_download(repo_id="zhengchong/CatVTON")

# CatVTON 파이프라인 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
base_pipeline = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt="zhengchong/CatVTON",
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype("fp16"),
    use_tf32=True,
    device=device,
    skip_safety_check=True,
)

# LoRA 가중치 및 파이프라인 캐싱
lora_ckpt_base = "lora_weights"
os.makedirs(lora_ckpt_base, exist_ok=True)
repo_id = "Coldbrew9/wheel-CatVTON"

# 기본 LoRA 가중치 로드
lora_filename_default = "best_loss_lora_r64_ep22_20250221_081824.pt"
lora_weights_path_default = hf_hub_download(
    repo_id=repo_id,
    filename=lora_filename_default,
    local_dir=lora_ckpt_base,
    repo_type="model"
)
logger.info(f"Downloaded default LoRA weights from {repo_id} to {lora_weights_path_default}")

# LoRA 설정 (기본값)
filename_base = os.path.basename(lora_filename_default)
match = re.search(r"lora_r(\d+)", filename_base)
lora_rank = int(match.group(1)) if match else 4
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_rank * 2,
    lora_dropout=0.1,
    target_modules=["to_q", "to_k", "to_v"],
)

# 캐싱된 파이프라인 딕셔너리
cached_pipelines: Dict[str, CatVTONPipeline] = {}

# LoRA 없는 기본 파이프라인
pipeline_no_lora = base_pipeline

# 기본 LoRA 파이프라인 캐싱
pipeline_with_lora = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt="zhengchong/CatVTON",
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype("fp16"),
    use_tf32=True,
    device=device,
    skip_safety_check=True,
)
pipeline_with_lora.unet = get_peft_model(pipeline_with_lora.unet, lora_config)
pipeline_with_lora.unet.load_state_dict(torch.load(lora_weights_path_default, map_location=device), strict=False)
logger.info(f"Loaded default LoRA weights into pipeline from {lora_weights_path_default}")
cached_pipelines["default"] = pipeline_with_lora
cached_pipelines["none"] = pipeline_no_lora

# LoRA 가중치 로드 및 캐싱 함수
def load_and_cache_lora_weights(lora_filename: str) -> CatVTONPipeline:
    if lora_filename in cached_pipelines:
        logger.info(f"Using cached pipeline for LoRA weight: {lora_filename}")
        return cached_pipelines[lora_filename]

    lora_weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=lora_filename,
        local_dir=lora_ckpt_base,
        repo_type="model"
    )
    logger.info(f"Downloaded LoRA weights: {lora_filename} to {lora_weights_path}")
    
    # 새 파이프라인 생성 및 LoRA 적용
    new_pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt="zhengchong/CatVTON",
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype("fp16"),
        use_tf32=True,
        device=device,
        skip_safety_check=True,
    )
    match = re.search(r"lora_r(\d+)", lora_filename)
    lora_rank = int(match.group(1)) if match else 4
    lora_config_dynamic = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.1,
        target_modules=["to_q", "to_k", "to_v"],
    )
    new_pipeline.unet = get_peft_model(new_pipeline.unet, lora_config_dynamic)
    new_pipeline.unet.load_state_dict(torch.load(lora_weights_path, map_location=device), strict=False)
    logger.info(f"Loaded dynamic LoRA weights into pipeline from {lora_weights_path}")

    # 캐싱
    cached_pipelines[lora_filename] = new_pipeline
    return new_pipeline

mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=device
)

# 로컬 저장 폴더
os.makedirs("tryon-images", exist_ok=True)

class TryonExperimentRequest(BaseModel):
    cloth_type: str  # 필수 필드
    lora_weight_name: Optional[str] = None  # LoRA 가중치 이름 (선택 사항, 기본값 None으로 기본 LoRA 사용)

# 새로운 엔드포인트: LoRA 실험을 위한 캐싱된 파이프라인 사용
@app.post("/tryon-experiment")
async def tryon_experiment(
    body_image: UploadFile = File(...),  # 필수
    cloth_image: UploadFile = File(...),  # 필수
    cloth_type: str = Form(...),         # 필수
    lora_weight_name: Optional[str] = Form(None)  # LoRA 가중치 이름 (선택 사항, 기본값 None으로 기본 LoRA 사용)
):
    logger.info(f"Processing tryon-experiment request for cloth_type: {cloth_type}, lora_weight_name: {lora_weight_name}")

    # 업로드된 이미지 처리
    person_image = Image.open(BytesIO(await body_image.read())).convert("RGB")
    cloth_image = Image.open(BytesIO(await cloth_image.read())).convert("RGB")
    logger.info("Loaded and converted uploaded images for experiment")

    # 파이프라인 선택 (캐싱된 파이프라인 사용)
    if lora_weight_name == "none":
        pipeline = cached_pipelines["none"]
    elif lora_weight_name:
        pipeline = load_and_cache_lora_weights(lora_weight_name)
    else:  # 기본 LoRA 사용
        pipeline = cached_pipelines["default"]

    # 추론
    with torch.no_grad():
        person_image = ImageOps.exif_transpose(person_image)
        cloth_image = ImageOps.exif_transpose(cloth_image)
        person_image = resize_and_crop(person_image, (768, 1024))
        cloth_image = resize_and_padding(cloth_image, (768, 1024))
        logger.info("Resized and cropped uploaded images for experiment inference")

        mask = automasker(person_image, mask_type=cloth_type)['mask']

        generator = torch.Generator(device=device).manual_seed(555)
        logger.info("Starting inference with CatVTON pipeline for experiment")
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

    # 결과 이미지를 메모리에 저장 후 스트리밍 응답
    result_image = result
    img_byte_arr = BytesIO()
    result_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    logger.info("Prepared streaming response with experiment result")

    return StreamingResponse(img_byte_arr, media_type="image/png", headers={"Lora-Weight-Used": lora_weight_name or "default"})

def download_image(url: str, save_path: str) -> str:
    try:
        logger.info(f"Downloading image from {url} to {save_path}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img.save(save_path)
        logger.info(f"Successfully downloaded and saved image to {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application with Uvicorn server for experimental try-on")
    uvicorn.run(app, host="0.0.0.0", port=8000)