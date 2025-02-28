# NVIDIA CUDA 12.8과 Python 3.10을 위한 베이스 이미지
# gcp를 위한 .env/web-project-438308-a8f3849fdf23.json 파일이 있는 디렉토리에 실행
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 Git, Python 3.10 설치
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10을 기본 Python으로 설정
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python
    
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# 종속성 설치 (requirements.txt 기반)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# VTON-PROJECT 리포지토리 클론
RUN git clone https://github.com/WheelWear/VTON-project.git 
# CatVTON 서브모듈 리포지토리 클론 (VTON-PROJECT 내부)
WORKDIR /app/VTON-project

RUN git clone https://github.com/Zheng-Chong/CatVTON.git
# pipeline_train.py 파일을 CatVTON/model 디렉토리로 복사
RUN cp pipeline_train.py CatVTON/model/

# 작업 디렉토리 복귀

# FastAPI 실행 (app.py를 사용하여 실행)
COPY .env/web-project-438308-a8f3849fdf23.json .env/web-project-438308-a8f3849fdf23.json
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# 메타데이터 추가 (선택 사항)
LABEL maintainer="Your Name <shchoi8687@gmail.com>"
LABEL description="CUDA-enabled FastAPI application for Wheelwear-VTON project"

# 도커 실행
# docker build -t testaiserver .
# docker run -itd -p 5555:5555 -v "/c/Users/009/Desktop/WheelWear/testAIserver:/app" --name aiserver testaiserver


# 도커 허브/azure 컨테이너 허브브에 이미지 업로드
# docker login or azr login
# docker tag testaiserver:v1.0 ditlswlwhs3/testaiserver:v1.0
# docker image push ditlswlwhs3/testaiserver:v1.0

# gcp 키
# GOOGLE_APPLICATION_CREDENTIALS ="./web-project-438308-a8f3849fdf23.json"
