import sys
import os
import io
import torch
import numpy as np
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from huggingface_hub import snapshot_download
from diffusers.image_processor import VaeImageProcessor

# CatVTON 디렉토리를 sys.path에 추가하여 CatVTON/app.py와 동일하게 모듈 임포트
catvton_dir = os.path.join(os.path.dirname(__file__), "CatVTON")
if catvton_dir not in sys.path:
    sys.path.insert(0, catvton_dir)

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# 추가: image_grid 함수 구현 (간단한 버전)
def image_grid(images, rows, cols):
    if not images:
        return None
    # 모든 이미지의 최대 width와 height 계산
    widths, heights = zip(*(i.size for i in images))
    max_width, max_height = max(widths), max(heights)
    grid_img = Image.new('RGB', (cols * max_width, rows * max_height))
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid_img.paste(img, (col * max_width, row * max_height))
    return grid_img

# 기본 설정값 (args 대체)
class Args:
    width = 768
    height = 1024
    output_dir = "./output"

args = Args()
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# repo_path 설정: attn_ckpt와 AutoMasker에서 사용
repo_path = snapshot_download(repo_id="zhengchong/CatVTON")

app = FastAPI()

# pipeline.py를 수정하지 않으므로 resume_path를 사용하지 않고,
# attn_ckpt, attn_ckpt_version, weight_dtype, use_tf32 등을 포함하여 파이프라인 인스턴스 생성
pipeline = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt="zhengchong/CatVTON",        # 학습된 try-on 모델 체크포인트 경로 (attn_ckpt로 사용)
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype("bf16"),
    use_tf32=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# AutoMasker 및 mask_processor 설정
mask_processor = VaeImageProcessor(
    vae_scale_factor=8, 
    do_normalize=False, 
    do_binarize=True, 
    do_convert_grayscale=True
)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda'
)

# def submit_function(
#     person_image,  # dict type: {"background": <path>, "layers": [<mask_path>, ...]}
#     cloth_image,   # path to cloth image
#     cloth_type,
#     num_inference_steps,
#     guidance_scale,
#     seed,
#     show_type
# ):
#     # Process person image and mask input
#     person_img_path = person_image["background"]
#     mask_path = person_image["layers"][0]
#     mask = Image.open(mask_path).convert("L")
#     if len(np.unique(np.array(mask))) == 1:
#         mask = None
#     else:
#         mask_np = np.array(mask)
#         mask_np[mask_np > 0] = 255
#         mask = Image.fromarray(mask_np)

#     # 결과 저장 경로 생성
#     date_str = datetime.now().strftime("%Y%m%d%H%M%S")
#     folder_path = os.path.join(args.output_dir, date_str[:8])
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     result_save_path = os.path.join(folder_path, date_str[8:] + ".png")

#     # Seed 설정
#     generator = None
#     if seed != -1:
#         generator = torch.Generator(device='cuda').manual_seed(seed)

#     # 이미지 로드 및 전처리
#     person_img = Image.open(person_img_path).convert("RGB")
#     cloth_img = Image.open(cloth_image).convert("RGB")
#     person_img = resize_and_crop(person_img, (args.width, args.height))
#     cloth_img = resize_and_padding(cloth_img, (args.width, args.height))
    
#     # 마스크 처리: mask가 None이면 automasker로 생성, 아니면 리사이즈 후 blur 처리
#     if mask is not None:
#         mask = resize_and_crop(mask, (args.width, args.height))
#     else:
#         mask = automasker(person_img, cloth_type)['mask']
#     mask = mask_processor.blur(mask, blur_factor=9)

#     # Inference 호출
#     result_image = pipeline(
#         image=person_img,
#         condition_image=cloth_img,
#         mask=mask,
#         num_inference_steps=num_inference_steps,
#         guidance_scale=guidance_scale,
#         generator=generator
#     )[0]
    
#     # Post-process: 결과 이미지를 grid 형태로 합침
#     masked_person = vis_mask(person_img, mask)
#     save_result_image = image_grid([person_img, masked_person, cloth_img, result_image], 1, 4)
#     save_result_image.save(result_save_path)
    
#     if show_type == "result only":
#         return result_image
#     else:
#         width, height = person_img.size
#         if show_type == "input & result":
#             condition_width = width // 2
#             conditions = image_grid([person_img, cloth_img], 2, 1)
#         else:
#             condition_width = width // 3
#             conditions = image_grid([person_img, masked_person, cloth_img], 3, 1)
#         conditions = conditions.resize((condition_width, height), Image.NEAREST)
#         new_result_image = Image.new("RGB", (width + condition_width + 5, height))
#         new_result_image.paste(conditions, (0, 0))
#         new_result_image.paste(result_image, (condition_width + 5, 0))
#         return new_result_image


@app.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    cloth_type: str = Form(...),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(2.5),
    seed: int = Form(-1),
    show_type: str = Form("result only")
):
    """
    /tryon 엔드포인트:
    - 사람 이미지와 옷 이미지, 그리고 cloth_type(의상 타입)을 받아 try-on 결과 이미지를 생성하여 반환합니다.
    """
    try:
        # 업로드된 파일을 읽은 후 PIL.Image로 변환
        person_contents = await person_image.read()
        cloth_contents = await cloth_image.read()
        # FastAPI 환경에서는 파일 경로 대신 메모리 내 이미지 데이터를 사용
        person_img = Image.open(io.BytesIO(person_contents)).convert("RGB")
        cloth_img = Image.open(io.BytesIO(cloth_contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": "유효하지 않은 이미지 파일입니다.", "detail": str(e)}
        )

    try:
        # 자동 마스킹 수행
        # 여기서는 automasker를 통해 생성된 mask를 사용합니다.
        mask = automasker(person_img, cloth_type)['mask']
        mask = mask_processor.blur(mask, blur_factor=9)
        # Inference 호출
        result_img = pipeline(
            image=person_img,
            condition_image=cloth_img,
            mask=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=(torch.Generator(device='cuda').manual_seed(seed) if seed != -1 else None)
        )[0]
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "이미지 생성 중 오류가 발생했습니다.", "detail": str(e)}
        )

    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)