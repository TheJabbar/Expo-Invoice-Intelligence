import asyncio
import schedule
import time
import subprocess
import os
from datetime import datetime
from pathlib import Path
from loguru import logger
import threading


def run_retraining():
    """
    Run the retraining process using the latest weekly training data
    """
    try:
        logger.info("Starting weekly retraining process...")
        
        # Get the model directory and training data directory
        model_dir = Path(os.getenv("MODEL_DIR", "/app/ml/model"))
        train_data_dir = Path(os.getenv("TRAIN_DATA_DIR", "/app/ml/train"))
        
        # Find the most recent weekly training directory
        ml_dir = Path("/app/ml")
        weekly_dirs = [d for d in ml_dir.iterdir() if d.is_dir() and '_' in d.name and len(d.name) == 17]  # Format: DDMMYYYY_DDMMYYYY
        
        if not weekly_dirs:
            logger.warning("No weekly training directories found, skipping retraining")
            return False
        
        # Use the most recent weekly directory
        latest_weekly_dir = max(weekly_dirs, key=lambda x: x.name)
        train_txt_path = latest_weekly_dir / "train.txt"
        
        if not train_txt_path.exists():
            logger.warning(f"Training data file not found in {latest_weekly_dir}, skipping retraining")
            return False
        
        # Get the base model path
        model_path = model_dir / "easyocr_model.pth"  # Using EasyOCR model
        if not model_path.exists():
            logger.error(f"Base model not found: {model_path}")
            return False
        
        logger.info(f"Using training data from: {latest_weekly_dir}")
        logger.info(f"Using base model: {model_path}")
        
        # Run the training script
        train_script_path = Path("/app/ml/train_model.py")
        if not train_script_path.exists():
            logger.error(f"Training script not found: {train_script_path}")
            return False
        
        cmd = ["python", str(train_script_path)]
        
        logger.info(f"Running retraining command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd="/app"
        )
        
        logger.info("Retraining completed successfully")
        logger.info(result.stdout)
        
        # Record the retraining completion
        retrain_log_path = ml_dir / "retrain_history.log"
        with open(retrain_log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()}: Retraining completed using {latest_weekly_dir.name}\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        return False


def start_scheduler():
    """
    Start the scheduler to run retraining every Sunday
    """
    logger.info("Starting weekly retraining scheduler...")
    
    # Schedule retraining to run every Sunday at 2 AM
    schedule.every().sunday.at("02:00").do(run_retraining)
    
    # For testing purposes, we can also schedule it to run every minute
    # schedule.every().minute.do(run_retraining)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    # Run the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    logger.info("Scheduler started successfully")


def trigger_manual_retrain():
    """
    Trigger a manual retraining run
    """
    logger.info("Manual retraining triggered")
    return run_retraining()