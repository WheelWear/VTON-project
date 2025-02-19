#!/bin/bash
# train_lora.sh

# 에폭 수 직접 설정
num_epochs=20

echo "Starting all experiments with num_epochs=$num_epochs..."

# lr, lora_rank, accumulation_steps 조합으로 반복
for lr in 1e-4 5e-5 1e-5; do
    for lora_rank in 4 8 16 32; do
        for accum_steps in 4; do
            # 출력 디렉토리 동적으로 생성 (에폭 정보 포함)
            output_dir="./experiments/ckpt_lr_${lr}_rank_${lora_rank}_accum_${accum_steps}_ep_${num_epochs}"
            echo "Running with lr=$lr, lora_rank=$lora_rank, accumulation_steps=$accum_steps, num_epochs=$num_epochs..."

            # Python 명령어 실행
            python train_lora.py --data_root_path ./dataset --output_dir "$output_dir" --use_fp16 True --num_epochs "$num_epochs" --batch_size 1 --lr "$lr" --lora_rank "$lora_rank" --accumulation_steps "$accum_steps"
        done
    done
done

echo "All experiments finished!"