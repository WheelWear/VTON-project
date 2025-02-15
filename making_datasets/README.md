
### Data Preparation 
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


+---raw
|   +---cloth
|   +---not_team_data
|   \---person
+---renamed
|   +---renamed_cloth_images
|   \---renamed_person_images
\---roboflow
```