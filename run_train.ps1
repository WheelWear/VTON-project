# run_train.ps1

# Python 가상환경을 사용하는 경우, 가상환경 활성화 명령어 추가 가능
# 예: .\venv\Scripts\Activate.ps1

# 학습 스크립트 실행 명령어
python ./demo_train_use_pipe.py --data_root_path ./viton-hd --output_dir ./ckpt --num_epochs 2 --batch_size 1 --lr 1e-4 --lora_rank 2

python demo_train_use_pipe.py  --data_root_path /content --output_dir ./ckpt --use_fp16 True --num_epochs 2 --batch_size 1 --lr 1e-4 --lora_rank 2 --num_inference_steps 50