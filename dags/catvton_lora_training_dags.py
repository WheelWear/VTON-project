from airflow import DAG
from airflow.decorators import task
from airflow.utils.log.logging_mixin import LoggingMixin
from datetime import datetime, timedelta
import os
import sys
import torch
import wandb
import mlflow
import dagshub

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # dags 상위 디렉토리 추가
from train_for_airflow import (
    parse_args, Custom_VITONHDTrainDataset, Custom_VITONHDTestDataset,
    initialize_model, train_model, validate_model, save_and_upload_model
)

# Airflow 로거 설정
log = LoggingMixin().log

# DAG 정의
with DAG(
    dag_id='catvton_lora_training',
    description='CatVTON LoRA Fine-tuning Pipeline',
    schedule_interval=timedelta(days=7),  # 일주일 한 번 실행
    start_date=datetime(2025, 3, 20),
    catchup=False,
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }
) as dag:

    @task
    def prepare_data():
        """데이터셋 준비 태스크"""
        log.info("Starting prepare_data task")
        try:
            args = parse_args()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log.info(f"Device set to: {device}")

            # Wandb와 MLflow 초기화
            log.info("Initializing Wandb and MLflow")
            dagshub.init(repo_owner='ColdTbrew', repo_name='VTON-project', mlflow=True)
            wandb.init(project="VTON-project", config=vars(args))
            mlflow.set_experiment("CatVTON_LoRA_Training")
            run_name = os.path.join(args.output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.start_run(run_name=run_name)
            mlflow.log_params(vars(args))
            mlflow.pytorch.autolog()
            log.info(f"MLflow run started with name: {run_name}")

            log.info("prepare_data task completed successfully")
            return {"args": args, "device": device, "run_name": run_name}
        except Exception as e:
            log.error(f"Error in prepare_data task: {str(e)}")
            raise

    @task
    def train_model_task(data):
        """모델 학습 태스크"""
        log.info("Starting train_model_task")
        try:
            args = data["args"]
            device = data["device"]
            log.info("Loading training dataset")
            train_dataset = Custom_VITONHDTrainDataset(args)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            log.info("Training dataset loaded successfully")

            log.info("Initializing model")
            pipeline, model = initialize_model(args, device)
            log.info("Model initialized successfully")

            log.info("Starting model training")
            pipeline, model, best_loss, best_epoch = train_model(args, train_dataloader, pipeline, model, device)
            log.info(f"Model training completed. Best loss: {best_loss}, Best epoch: {best_epoch}")

            # 모델 저장
            model_path = os.path.join(args.output_dir, f"model_{dag.run_id}.pt")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            log.info(f"Model saved to: {model_path}")

            log.info("train_model_task completed successfully")
            return {
                "model_path": model_path,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "args": args,
                "device": device
            }
        except Exception as e:
            log.error(f"Error in train_model_task: {str(e)}")
            raise

    @task
    def validate_model_task(data, train_result, dag_run):
        """검증 태스크"""
        log.info("Starting validate_model_task")
        try:
            args = data["args"]
            device = data["device"]
            model_path = train_result["model_path"]
            best_epoch = train_result["best_epoch"]
            log.info(f"Loading model from: {model_path}")

            # 모델 로드
            pipeline, model = initialize_model(args, device)
            model.load_state_dict(torch.load(model_path))
            log.info("Model loaded successfully")

            log.info("Loading validation dataset")
            val_dataset = Custom_VITONHDTestDataset(args)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            log.info("Validation dataset loaded successfully")

            log.info("Starting model validation")
            avg_psnr, avg_ssim, avg_lpips = validate_model(args, val_dataloader, pipeline, model, device, best_epoch, dag_run.run_id)
            log.info(f"Validation completed. PSNR: {avg_psnr}, SSIM: {avg_ssim}, LPIPS: {avg_lpips}")

            log.info("validate_model_task completed successfully")
            return {
                "avg_psnr": avg_psnr,
                "avg_ssim": avg_ssim,
                "avg_lpips": avg_lpips,
                "model_path": model_path,
                "args": args
            }
        except Exception as e:
            log.error(f"Error in validate_model_task: {str(e)}")
            raise

    @task
    def save_and_upload_task(validate_result, dag_run):
        """모델 저장 및 업로드 태스크"""
        log.info("Starting save_and_upload_task")
        try:
            args = validate_result["args"]
            avg_lpips = validate_result["avg_lpips"]
            model_path = validate_result["model_path"]
            log.info(f"Loading model from: {model_path}")

            # 모델 로드
            _, model = initialize_model(args, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            model.load_state_dict(torch.load(model_path))
            log.info("Model loaded successfully")

            log.info("Saving and uploading model")
            best_lpips = float("inf")
            best_lpips = save_and_upload_model(args, model, avg_lpips, best_lpips, dag_run.run_id)
            log.info(f"Model saved and uploaded. Best LPIPS: {best_lpips}")

            # Wandb와 MLflow 종료
            log.info("Finishing Wandb and MLflow")
            wandb.finish()
            mlflow.end_run()

            log.info("save_and_upload_task completed successfully")
            return best_lpips
        except Exception as e:
            log.error(f"Error in save_and_upload_task: {str(e)}")
            raise

    # 태스크 실행 및 의존성 설정
    data = prepare_data()
    train_result = train_model_task(data)
    validate_result = validate_model_task(data, train_result)
    save_and_upload_task(validate_result)