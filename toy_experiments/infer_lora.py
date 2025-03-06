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

def process_tryon_experiment(
    body_image_path: str,  # 로컬 사람 이미지 파일 경로
    cloth_image_path: str,  # 로컬 옷 이미지 파일 경로
    cloth_type: str,       # 필수
    lora_weight_name: Optional[str] = None,  # LoRA 가중치 이름 (선택 사항)
    use_repaint = True
):
    # 로컬 이미지 파일 로드
    person_image = Image.open(body_image_path).convert("RGB")
    cloth_image = Image.open(cloth_image_path).convert("RGB")

    # 공통 이미지 전처리
    person_image = ImageOps.exif_transpose(person_image)
    cloth_image = ImageOps.exif_transpose(cloth_image)
    person_image = resize_and_crop(person_image, (768, 1024))
    cloth_image = resize_and_padding(cloth_image, (768, 1024))

    mask_base = automasker(person_image, mask_type=cloth_type, use_resize_mask=False)['mask']
    mask = automasker(person_image, mask_type=cloth_type, use_resize_mask=True)['mask']

    mask_base.save(os.path.join("final_test","result", f"mask_base.png"), format="PNG")
    mask.save(os.path.join("final_test","result", f"mask.png"), format="PNG")

    # 기본 파이프라인 생성 (LoRA 적용 X)
    base_ckpt_path = snapshot_download(repo_id="booksforcharlie/stable-diffusion-inpainting", local_dir="./base_ckpt")
    attn_ckpt_path = snapshot_download(repo_id="zhengchong/CatVTON", local_dir="./attn_ckpt")
    pipeline_no_lora = CatVTONPipeline(
        base_ckpt=base_ckpt_path,
        attn_ckpt=attn_ckpt_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype("fp16"),
        use_tf32=True,
        device=device,
        skip_safety_check=True,
    )
    # LoRA를 적용하지 않은 추론
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(555)
        result_no_lora = pipeline_no_lora(
            image=person_image,
            condition_image=cloth_image,
            mask=mask_base,
            num_inference_steps=50,
            guidance_scale=2.5,
            height=1024,
            width=768,
            generator=generator,
        )[0]
    del pipeline_no_lora

    if use_repaint:
        result_no_lora = repaint(person_image, mask, result_no_lora)

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
    # state_dict = torch.load(lora_weights_path, map_location="cpu")
    # print(state_dict.keys())
    # # 파라미터 봐보기
    # pipeline_with_lora.unet.print_trainable_parameters()
    # print("norm ",state_dict["base_model.model.mid_block.attentions.0.transformer_blocks.0.attn2.to_v.lora_B.weight"].norm())  # 가중치 값이 0에 가까운지 확인
    # LoRA 적용 추론
    with torch.no_grad():
        result_with_lora = pipeline_with_lora(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=50,
            guidance_scale=2.5,
            height=1024,
            width=768,
            generator=generator,
        )[0]
    
    if use_repaint:
        result_with_lora = repaint(person_image, mask, result_with_lora)

    # 두 결과를 width 방향으로 붙이기
    combined_image = Image.new('RGB', (result_no_lora.width * 2, result_no_lora.height))
    combined_image.paste(result_no_lora, (0, 0))
    combined_image.paste(result_with_lora, (result_no_lora.width, 0))
    
    # # 차영상 계산
    # diff_image = Image.new('RGB', (result_no_lora.width, result_no_lora.height))
    # for x in range(result_no_lora.width):
    #     for y in range(result_no_lora.height):
    #         r1, g1, b1 = result_no_lora.getpixel((x, y))
    #         r2, g2, b2 = result_with_lora.getpixel((x, y))
    #         diff_image.putpixel((x, y), (abs(r1 - r2), abs(g1 - g2), abs(b1 - b2)))

    # # 차영상의 합 계산 및 출력
    # diff_sum = sum(sum(diff_image.getpixel((x, y))) for x in range(diff_image.width) for y in range(diff_image.height))
    # print(f"Sum of difference: {diff_sum}")

    # 결과 저장
    output_prefix = f"{body_image_path.split('/')[-1]}_{cloth_image_path.split('/')[-1].split('.')[0]}_real_final"

    result_no_lora.save(os.path.join("final_test","result", f"{output_prefix}_CatVTON.png"), format="PNG")
    result_with_lora.save(os.path.join("final_test","result", f"{output_prefix}_Ours.png"), format="PNG")
    combined_image.save(os.path.join("final_test","result", f"{output_prefix}_combined.png"), format="PNG")
    # diff_image.save(os.path.join("final_test","result", f"{output_prefix}_diff.png"), format="PNG")
    # result_with_lora.save(os.path.join("final_test","result", f"{output_prefix}_with_lora.png"), format="PNG")

    return os.path.join("tryon-images", f"{output_prefix}_combined.png")

if __name__ == "__main__":
    body_image_path = "C:/Users/coldbrew/VTON-project/final_test/person/people_1.jpg"
    cloth_image_path ="C:/Users/coldbrew/VTON-project/final_test/cloth_u/hr_upper_09.jpg"
    cloth_type = "upper"
    lora_weight_name = "Wheel-wear_lora_r16_ep10.pt"
    use_repaint = False
    # lora_weight_name = "best_lpips_lora_r32_lr1e-05_ep35_20250224_155113.pt"
    result_path = process_tryon_experiment(body_image_path, cloth_image_path, cloth_type, lora_weight_name, use_repaint)
    print(f"Result saved at: {result_path}")