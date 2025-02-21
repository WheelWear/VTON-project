FROM python:3.10

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY app.py .


# FastAPI 실행 시 모든 IP에서 접근 가능하도록 설정
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5555", "--reload"]

# 도커 실행
# docker build -t testaiserver .
# docker run -itd -p 5555:5555 -v "/c/Users/009/Desktop/WheelWear/testAIserver:/app" --name aiserver testaiserver


# 도커 허브/azure 컨테이너 허브브에 이미지 업로드
# docker login or azr login
# docker tag testaiserver:v1.0 ditlswlwhs3/testaiserver:v1.0
# docker image push ditlswlwhs3/testaiserver:v1.0

# gcp 키
# GOOGLE_APPLICATION_CREDENTIALS ="./web-project-438308-a8f3849fdf23.json"
