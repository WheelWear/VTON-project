from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from io import BytesIO
from PIL import Image, ExifTags, ImageOps
from typing import Optional
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

os.makedirs("logs", exist_ok=True)
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
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

# for recommend_size
from llm_agent import recommend_size

# 입력 데이터 모델 정의
class InputData(BaseModel):
    brand: str
    cloth_size: str
    cloth_type: str
    gender: str
    chest_circumference: float
    shoulder_width: float
    arm_length: float
    waist_circumference: float

# 출력 데이터 모델 정의
class OutputData(BaseModel):
    recommend_size: str
    additional_explanation: str
    references: list
    reference_num: int

# GCS 설정
credentials = service_account.Credentials.from_service_account_file('./.env/web-project-438308-a8f3849fdf23.json')
client = storage.Client(credentials=credentials, project='web-project-438308')
BUCKET_NAME = "wheelwear-bucket"

app = FastAPI()
count = 0
repo_path = snapshot_download(repo_id="zhengchong/CatVTON")

# CatVTON 파이프라인 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt="zhengchong/CatVTON",
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype("fp16"),
    use_tf32=True,
    device=device,
    skip_safety_check=True,
)
# Hugging Face에서 LoRA 가중치 다운로드
lora_ckpt = "lora_weights"
os.makedirs(lora_ckpt, exist_ok=True)
repo_id = "Coldbrew9/wheel-CatVTON"
lora_filename = "best_loss_lora_r64_ep22_20250221_081824.pt"
lora_weights_path = hf_hub_download(
    repo_id=repo_id,
    filename=lora_filename,
    local_dir=lora_ckpt,
    repo_type="model"
)
logger.info(f"Downloaded LoRA weights from {repo_id} to {lora_weights_path}")

filename_base = os.path.basename(lora_filename)
match = re.search(r"lora_r(\d+)", filename_base)
lora_rank = int(match.group(1)) if match else 4
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_rank * 2,
    lora_dropout=0.1,
    target_modules=["to_q", "to_k", "to_v"],
)
pipeline.unet = get_peft_model(pipeline.unet, lora_config)
pipeline.unet.load_state_dict(torch.load(lora_weights_path, map_location=device), strict=False)
logger.info(f"Loaded LoRA weights into pipeline from {lora_weights_path}")

mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=device
)

# 로컬 저장 폴더
os.makedirs("tryon-images", exist_ok=True)

class TryonRequest(BaseModel):
    top_cloth_url: Optional[HttpUrl] = None
    bottom_cloth_url: Optional[HttpUrl] = None
    dress_image_url: Optional[HttpUrl] = None
    body_image_url: HttpUrl  # 필수 필드
    cloth_type: str  # 필수 필드

@app.post("/tryon")
async def tryon(request: TryonRequest):
    global count
    count += 1
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    img_filename = f"vtonimage/generated_image{count}_{timestamp}.png"

    logger.info(f"Processing tryon request for body_image_url: {request.body_image_url}, cloth_type: {request.cloth_type}")

    # 이미지 다운로드
    person_path = download_image(request.body_image_url, "tryon-images/person.jpg")
    logger.info(f"Downloaded person image to {person_path}")
    cloth_path = None
    cloth_type = request.cloth_type
    if request.top_cloth_url and cloth_type == "upper":
        cloth_path = download_image(request.top_cloth_url, "tryon-images/top_cloth.jpg")
        logger.info(f"Downloaded top cloth image to {cloth_path}")
    elif request.bottom_cloth_url and cloth_type == "lower":
        cloth_path = download_image(request.bottom_cloth_url, "tryon-images/bottom_cloth.jpg")
        logger.info(f"Downloaded bottom cloth image to {cloth_path}")
    elif request.dress_image_url and cloth_type == "overall":
        cloth_path = download_image(request.dress_image_url, "tryon-images/overall.jpg")
        logger.info(f"Downloaded overall image to {cloth_path}")
    else:
        logger.error(f"cloth_type '{cloth_type}'에 맞는 cloth URL이 제공되지 않았습니다.")
        raise ValueError(f"cloth_type '{cloth_type}'에 맞는 cloth URL(top_cloth_url, bottom_cloth_url, dress_image_url)이 제공되지 않았습니다.")

    # 추론
    with torch.no_grad():
        person_image = Image.open(person_path).convert("RGB")
        cloth_image = Image.open(cloth_path).convert("RGB")
        person_image = ImageOps.exif_transpose(person_image)
        cloth_image = ImageOps.exif_transpose(cloth_image)
        person_image = resize_and_crop(person_image, (768, 1024))
        cloth_image = resize_and_padding(cloth_image, (768, 1024))
        person_image.save("tryon-images/person_resized.jpg")
        cloth_image.save("tryon-images/cloth_resized.jpg")
        logger.info("Resized and cropped images for inference")

        mask = automasker(person_image, mask_type=cloth_type)['mask']
        
        generator = torch.Generator(device=device).manual_seed(555)
        logger.info("Starting inference with CatVTON pipeline")
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

    # GCS 업로드
    result_image = result
    result_image.save("tryon-images/result.png")
    logger.info("Saved inference result to tryon-images/result.png")
    img_byte_arr = BytesIO()
    result_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(img_filename)
    blob.upload_from_file(img_byte_arr, content_type="image/png")
    image_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{img_filename}"
    logger.info(f"Uploaded result image to GCS: {image_url}")

    return {"result_image_url": image_url}

@app.post("/tryon-upload")
async def tryon_upload(
    body_image: UploadFile = File(...),  # 필수
    cloth_image: UploadFile = File(...),  # 필수
    cloth_type: str = Form(...)          # 필수
):
    logger.info(f"Processing tryon-upload request for cloth_type: {cloth_type}")

    # 업로드된 이미지 처리
    person_image = Image.open(BytesIO(await body_image.read())).convert("RGB")
    cloth_image = Image.open(BytesIO(await cloth_image.read())).convert("RGB")
    logger.info("Loaded and converted uploaded images")

    # 추론
    with torch.no_grad():
        person_image = ImageOps.exif_transpose(person_image)
        cloth_image = ImageOps.exif_transpose(cloth_image)
        person_image = resize_and_crop(person_image, (768, 1024))
        cloth_image = resize_and_padding(cloth_image, (768, 1024))
        logger.info("Resized and cropped uploaded images for inference")

        # automasker는 PIL 이미지를 처리한다고 가정 (필요 시 경로로 변환)
        mask = automasker(person_image, mask_type=cloth_type)['mask']

        generator = torch.Generator(device=device).manual_seed(555)
        logger.info("Starting inference with CatVTON pipeline for upload")
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
    logger.info("Prepared streaming response with inference result")

    return StreamingResponse(img_byte_arr, media_type="image/png")

# 사이즈 추천 엔드포인트
@app.post("/recommend_size", response_model=OutputData)
async def recommend_size_endpoint(data: InputData):
    """
    사용자의 데이터를 받아 agent.py의 recommend_size 함수로 사이즈를 추천받아.
    """
    result = recommend_size(data)
    return result

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
    logger.info("Starting FastAPI application with Uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)