import sys
import os
import io
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image

# CatVTON 디렉토리를 sys.path에 추가하여 CatVTON/app.py와 동일하게 모듈 임포트
catvton_dir = os.path.join(os.path.dirname(__file__), "CatVTON")
if catvton_dir not in sys.path:
    sys.path.insert(0, catvton_dir)

from pipeline import CatVTONPipeline
from utils import init_weight_dtype

app = FastAPI()

# CatVTON/app.py의 인자 구성과 유사하게 파이프라인 인스턴스 생성
# mixed precision은 bf16 모드로 작동하도록 weight_dtype을 bf16으로 설정합니다.
pipeline = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    resume_path="zhengchong/CatVTON",  # 학습된 tryon 모델 체크포인트 경로
    weight_dtype=init_weight_dtype("bf16"),
    use_tf32=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

@app.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    cloth_type: str = Form(...)
):
    """
    /tryon 엔드포인트:
    - 사람 이미지와 옷 이미지, cloth_type(의상 타입)을 받아 try-on 결과 이미지를 생성하여 반환합니다.
    """
    try:
        # 업로드된 파일들을 읽고 PIL.Image로 변환
        person_contents = await person_image.read()
        cloth_contents = await cloth_image.read()
        person_img = Image.open(io.BytesIO(person_contents)).convert("RGB")
        cloth_img = Image.open(io.BytesIO(cloth_contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": "유효하지 않은 이미지 파일입니다.", "detail": str(e)}
        )

    try:
        # CatVTONPipeline 호출 (cloth_type 인자 포함)
        result_img = pipeline(person_img, cloth_img, cloth_type)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "이미지 생성 중 오류가 발생했습니다.", "detail": str(e)}
        )

    # 결과 이미지를 PNG 포맷으로 스트리밍 응답 반환
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)