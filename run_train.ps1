# run_train.ps1

# 학습 스크립트 실행 명령어

# local 실행 데이터셋 경로 바꾸시오
python ./demo_train_use_pipe.py --data_root_path ./viton-hd --output_dir ./ckpt --use_fp16 True --num_epochs 10 --batch_size 1 --lr 1e-4 --lora_rank 4 --num_inference_steps 999
# 코랩환경 실행
python demo_train_use_pipe.py  --data_root_path /content --output_dir ./ckpt --use_fp16 False --num_epochs 5 --batch_size 1 --lr 1e-4 --lora_rank 4 --num_inference_steps 999
# 스탭 2
python demo_train_use_pipe.py  --data_root_path ./viton-hd --output_dir ./ckpt --use_fp16 True --num_epochs 5 --batch_size 1 --lr 1e-4 --lora_rank 4 --num_inference_steps 2 --use_tf32 False

#for 1 timestep
python train_lora.py  --data_root_path ./dataset --output_dir ./experiments/ckpt --use_fp16 True --num_epochs 5 --batch_size 1 --lr 1e-4 --lora_rank 4