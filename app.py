from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import io
import torch

from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype  # if needed for weight_dtype

app = FastAPI()

# 파이프라인 인스턴스 생성 (필요한 경로와 옵션들은 필요에 맞게 수정)
pipeline = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt="PATH_TO_ATTN_CKPT",  # 실제 attn_ckpt 경로 또는 repo_path를 지정
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype("fp32"),
    use_tf32=False,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

@app.post("/tryon")
async def tryon(image: UploadFile = File(...), cloth_type: str = Form(...)):
    """
    /tryon 엔드포인트:
    - 이미지 파일과 cloth_type(의상 타입)을 받아서 try-on 결과 이미지를 생성하여 반환합니다.
    """
    try:
        # 업로드된 파일을 PIL.Image로 변환
        contents = await image.read()
        person_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "유효하지 않은 이미지 파일입니다."})

    try:
        # 파이프라인 실행 (실제 파이프라인 호출 방식에 맞춰 인자 전달)
        # 참고: 실제 CatVTONPipeline이 person_img와 cloth_type을 인자로 처리할 수 있도록 구현되어 있어야 합니다.
        result_img = pipeline(person_img, cloth_type)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "이미지 생성 중 오류가 발생했습니다.", "details": str(e)})

    # 결과 이미지를 PNG 포맷으로 반환
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)