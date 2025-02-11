import argparse
import os
import glob
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# CatVTON 디렉토리를 sys.path에 추가하여 CatVTON/app.py와 동일하게 모듈 임포트
catvton_dir = os.path.join(os.path.dirname(__file__), "CatVTON")
if catvton_dir not in sys.path:
    sys.path.insert(0, catvton_dir)

from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# 사용자 정의 데이터셋 (사람 이미지, 옷 이미지, 마스크 이미지)
class CatVTONDataset(Dataset):
    def __init__(self, person_dir, cloth_dir, mask_dir, transform=None):
        self.person_paths = sorted(glob.glob(os.path.join(person_dir, "*.png")))
        self.cloth_paths = sorted(glob.glob(os.path.join(cloth_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.person_paths)

    def __getitem__(self, idx):
        person = Image.open(self.person_paths[idx]).convert("RGB")
        cloth = Image.open(self.cloth_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            person = self.transform(person)
            cloth = self.transform(cloth)
            mask = self.transform(mask)
        return person, cloth, mask

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = CatVTONPipeline(
        base_ckpt=args.base_model_path,
        attn_ckpt=args.attn_ckpt,
        attn_ckpt_version=args.attn_ckpt_version,
        weight_dtype=init_weight_dtype(args.mixed_precision),
        use_tf32=args.allow_tf32,
        device=device
    )
    model = pipeline.unet
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()  # 예시용 손실 함수 (필요에 따라 교체)

    transform = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor()
    ])
    dataset = CatVTONDataset(args.person_dir, args.cloth_dir, args.mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for person, cloth, mask in dataloader:
            person = person.to(device)
            cloth = cloth.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()

            # Forward pass: pipeline.__call__는 (image, condition_image, mask, ...) 형태로 호출
            output = pipeline(
                image=person,
                condition_image=cloth,
                mask=mask,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=None  # 필요 시 torch.Generator로 시드 설정 가능
            )[0]

            # 예시로 사람 이미지를 target으로 MSE 손실 계산 (실제 손실 함수는 목적에 맞게 수정)
            loss = loss_fn(output, person)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}")

    # 학습 완료 후 모델 저장
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"catvton_unet_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Stable Diffusion base model checkpoint path")
    parser.add_argument("--attn_ckpt", type=str, required=True,
                        help="Attention module checkpoint path or repo id")
    parser.add_argument("--attn_ckpt_version", type=str, default="mix",
                        help="Attention checkpoint version (e.g., mix, vitonhd, dresscode)")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        help="Mixed precision mode (bf16, fp16, etc.)")
    parser.add_argument("--allow_tf32", action="store_true",
                        help="Enable TF32 for faster matmul on supported GPUs")
    parser.add_argument("--person_dir", type=str, required=True,
                        help="Directory containing person images (.png)")
    parser.add_argument("--cloth_dir", type=str, required=True,
                        help="Directory containing cloth images (.png)")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Directory containing mask images (.png)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    main(args)