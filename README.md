# VTON-project


## 파인튜닝을 위한 데이터셋 구축
```
dataset/
├── cloth/
│   ├── lower_img/       # 하의 이미지 저장 (예: 모델의 하의 착용 이미지)
│   │   └── 00000.jpg    # 예시 이미지 파일
│   ├── lower_mask/      # 하의 마스크 이미지 저장 (예: 세분화 결과)
│   │   └── 00000.jpg
│   ├── upper_img/       # 상의 이미지 저장 (예: 모델의 상의 착용 이미지)
│   │   └── 00000.jpg
│   └── upper_mask/      # 상의 마스크 이미지 저장 (예: 세분화 결과)
│       └── 00000.jpg
├── image/               # 원본 이미지 저장 (전체 신체 혹은 기타 이미지)
│   └── 00000.jpg
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

- **cloth_upper_sample / cloth_lower_sample**: 옷(상의, 하의)의 실제 이미지  
- **cloth_upper_mask_sample / cloth_lower_mask_sample**: 해당 옷 영역을 마스킹한 이미지  
- **person_sample**: 착용자가 포함된 전체 신체 이미지  
- **person_lower_mask_sample / person_upper_mask_sample**: 인물의 하반신 및 상반신에 대한 마스크 이미지  
