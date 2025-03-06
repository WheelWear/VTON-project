# run_hyper_train.ps1

# 에폭 수 직접 설정
$num_epochs = 5
$filename = "train_lora_sh.py"

# 실험 시작 메시지
Write-Host "Starting all experiments with num_epochs=$num_epochs..."

# lr, lora_rank, accumulation_steps 조합으로 반복
foreach ($lr in @("1e-4", "1e-5")) {
    foreach ($lora_rank in @(4, 8, 16, 32)) {
        foreach ($accum_steps in @(4, 8, 16)) {
            # 출력 디렉토리 동적으로 생성 (에폭 정보 포함)
            $output_dir = "./experiments/ckpt_lr_${lr}_rank_${lora_rank}_accum_${accum_steps}_ep_${num_epochs}"
            Write-Host "Running with lr=$lr, lora_rank=$lora_rank, accumulation_steps=$accum_steps, num_epochs=$num_epochs..."
            
            # Python 명령어 실행
            $command = "python $filename --data_root_path ./dataset --output_dir $output_dir --use_fp16 True --num_epochs $num_epochs --batch_size 1 --lr $lr --lora_rank $lora_rank --accumulation_steps $accum_steps"
            Invoke-Expression $command
        }
    }
}

Write-Host "All experiments finished!"