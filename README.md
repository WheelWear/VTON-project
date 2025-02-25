# VTON-project

### llm_agent.py
- recommend_size  엔드포인트


input ex.
```{
  "brand": "나이키",
  "cloth_size": "100",
  "cloth_type" : "top",
  "gender": "남성",
  "chest_circumference": 32,
  "shoulder_width": 55,
  "arm_length": 42,
  "waist_circumference": 14
}
```

output ex.
```
{
  "recommend_size": "M",
  "additional_explanation": "최근 나이키 의류가 이전보다 크게 제작된다는 리뷰가 많으며, 사용자의 일반적인 사이즈(100)와 비교했을 때 M 사이즈가 적절할 것으로 판단됩니다.",
  "references": [
    "https://www.reddit.com/r/Nike/comments/15luox2/why_does_nike_keep_making_their_shirts_bigger/",
    "https://www.trustpilot.com/review/www.nike.com",
    "https://www.today.com/shop/nike-go-firm-support-leggings-review-rcna190644"
  ],
  "reference_num": 3
}
```


input type, output type
```
class InputData(BaseModel):
    brand: str
    cloth_size: str
    cloth_type: str
    gender: str
    chest_circumference: float
    shoulder_width: float
    arm_length: float
    waist_circumference: float

class OutputData(BaseModel):
    recommend_size: str
    additional_explanation: str
    references: list
    reference_num: int
```

## Docker
```
#docker pull base image
docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
#docker build
docker build -t coldbrew9/wheelwear-cu12.4-p3.10:latest .
#docker run
docker run --gpus all -it -p 8000:8000 -v "$($PWD.Path):/app" coldbrew9/wheelwear-cu12.4-p3.10:latest
```

## Run Train
```
# local train
python train_lora.py  --data_root_path ./dataset --output_dir ./experiments/ckpt --use_fp16 True --num_epochs 5 --batch_size 1 --lr 1e-4 --lora_rank 4 --accumulation_steps 4

# 코랩환경 실행
python train_lora.py  --data_root_path /content --output_dir ./experiments/ckpt --use_fp16 True --num_epochs 5 --batch_size 1 --lr 1e-4 --lora_rank 4
```

## Prepare dataset
- unzip and rename
```
unzip dataset_v{version}.zip .
mv dataset_v{version} dataset
```
## 파인튜닝을 위한 데이터셋 구축 (앉아있는 자세)
```
dataset/
├── cloth/
│   ├── lower_img/       
│   │   └── 00000.jpg    # 하의 이미지
│   ├── lower_mask/      
│   │   └── 00000.jpg    # 하의 이미지의 마스크
│   ├── upper_img/       
│   │   └── 00000.jpg    # 하의 이미지
│   └── upper_mask/      
│       └── 00000.jpg    # 하의 이미지의 마스크
├── image/               
│   └── 00000.jpg        # 사람 이미지지
├── image_mask_L/        # 이미지의 하반신 마스크 저장 (Lower 부분)
│   └── 00000.jpg
└── image_mask_U/        # 이미지의 상반신 마스크 저장 (Upper 부분)
    └── 00000.jpg
```

### 데이터셋 구성 예시시
| 분류                  | 원본 이미지 예시                 | 마스크 이미지 예시                          |
|-----------------------|----------------------------------|---------------------------------------------|
| **상의(Cloth Upper)** | <img src="resource/img/cloth_upper_sample.jpg" width="200" /> | <img src="resource/img/cloth_upper_mask_sample.jpg" width="200" /> |
| **하의(Cloth Lower)** | <img src="resource/img/cloth_lower_sample.jpg" width="200" /> | <img src="resource/img/cloth_lower_mask_sample.jpg" width="200" /> |
| **인물(Person)**      | <img src="resource/img/person_sample.jpg" width="200" />      | <table><tr><td>하반신 마스크:<br><img src="resource/img/person_lower_mask_sample.jpg" width="200" /></td><td>상반신 마스크:<br><img src="resource/img/person_upper_mask_sample.jpg" width="200" /></td></tr></table> |

