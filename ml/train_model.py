#!/usr/bin/env python3
"""
Training script for improving invoice processing with corrected data
Following the example structure to read from CSV training set
"""
import os
import yaml
from pathlib import Path
from loguru import logger
import pandas as pd
from datetime import datetime
import sys
import os
# Add the project root to the path to import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up two levels to project root
sys.path.insert(0, project_root)

try:
    from api.utils.model_version_manager import increment_model_version
except ImportError:
    # If import fails, define a dummy function for testing
    def increment_model_version(description=""):
        print(f"Model version increment called with: {description}")
        return "v1.0-EasyOCR"


def get_config(config_path):
    """
    Load configuration from YAML file and process character set from CSV data
    """
    with open(config_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    
    # Process character set from training data if needed
    if opt.get('lang_char') == 'None':
        characters = ''
        train_data_dir = opt.get('train_data', os.getenv("TRAIN_DATA_DIR", "./ml"))
        
        # Find weekly directories with labels.csv
        ml_dir = Path(train_data_dir)
        if not ml_dir.exists():
            ml_dir = Path("./ml")  # Fallback to relative path
        weekly_dirs = [d for d in ml_dir.iterdir() if d.is_dir() and '_' in d.name and len(d.name) == 17]  # Format: DDMMYYYY_DDMMYYYY
        
        if weekly_dirs:
            # Use the most recent weekly directory
            latest_weekly_dir = max(weekly_dirs, key=lambda x: x.name)
            csv_path = latest_weekly_dir / 'labels.csv'
            
            if csv_path.exists():
                # Read CSV with the expected format
                df = pd.read_csv(csv_path, usecols=['filename', 'words'], keep_default_na=False)
                all_char = ''.join(df['words'].astype(str))
                characters += ''.join(set(all_char))
        
        characters = sorted(set(characters))
        opt['character'] = ''.join(characters)
    else:
        opt['character'] = opt.get('number', '') + opt.get('symbol', '') + opt.get('lang_char', '')
    
    # Create experiment directory
    experiment_name = opt.get('experiment_name', 'invoice_ocr')
    os.makedirs(f'./saved_models/{experiment_name}', exist_ok=True)
    return opt


def train(opt, amp=False):
    """
    Run the real EasyOCR retraining process using the training data
    """
    # Find the most recent weekly training directory
    ml_dir = Path(opt.get('train_data', os.getenv("TRAIN_DATA_DIR", "./ml")))
    if not ml_dir.exists():
        ml_dir = Path("./ml")  # Fallback to relative path

    # Check both the main ml directory and the train subdirectory for weekly directories
    weekly_dirs = [d for d in ml_dir.iterdir() if d.is_dir() and '_' in d.name and len(d.name) == 17]  # Format: DDMMYYYY_DDMMYYYY

    # If no directories found in main ml directory, check the train subdirectory
    if not weekly_dirs:
        train_subdir = ml_dir / "train"
        if train_subdir.exists():
            weekly_dirs = [d for d in train_subdir.iterdir() if d.is_dir() and '_' in d.name and len(d.name) == 17]

    if not weekly_dirs:
        print("Skipping retraining: no weekly training directories found")
        return False

    # Use the most recent weekly directory
    latest_weekly_dir = max(weekly_dirs, key=lambda x: x.name)
    labels_csv_path = latest_weekly_dir / "labels.csv"

    if not labels_csv_path.exists():
        print(f"No labels.csv file in {latest_weekly_dir}, checking for legacy train.txt")
        train_txt_path = latest_weekly_dir / "train.txt"

        if not train_txt_path.exists():
            print(f"Skipping retraining: no labels.csv or train.txt file in {latest_weekly_dir}")
            return False
        else:
            # Convert legacy train.txt to labels.csv format
            print(f"Converting legacy train.txt to labels.csv format...")
            from api.utils.cropping.image_cropper import convert_train_txt_to_csv
            convert_train_txt_to_csv(train_txt_path, latest_weekly_dir)
            labels_csv_path = latest_weekly_dir / "labels.csv"

    try:
        # Read the training data
        df = pd.read_csv(labels_csv_path, usecols=['filename', 'words'], keep_default_na=False)
        sample_count = len(df)

        print(f"Found {sample_count} training samples in {labels_csv_path}")

        if sample_count < 5:  # Minimum samples needed
            print(f"Skipping retraining: only {sample_count} samples in {latest_weekly_dir}")
            return False

        print(f"Starting EasyOCR retraining with {sample_count} samples from {latest_weekly_dir.name}")

        # Actually run the retraining process
        from ml.retrain_easyocr import run_easyocr_retraining, prepare_training_data

        # Prepare the training data in EasyOCR format using the found directory
        train_data_dir = prepare_training_data(latest_weekly_dir)
        if not train_data_dir:
            print("Failed to prepare training data")
            return False

        # Run the actual retraining
        success = run_easyocr_retraining(train_data_dir)
        if not success:
            print("EasyOCR retraining failed")
            return False

        print(f"EasyOCR retraining completed with {sample_count} samples from {latest_weekly_dir.name}")

        # Increment model version after successful retraining
        new_version = increment_model_version(f"EasyOCR model retrained with {sample_count} samples from {latest_weekly_dir.name}")
        print(f"  Model version updated to: {new_version}")

        return True

    except Exception as e:
        print(f"Error during retraining: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_training():
    """
    Main training function that follows the example structure
    """
    logger.info("Processing corrected data to improve invoice processing accuracy")
    
    # Create a basic config file if it doesn't exist
    config_path = "config_files/invoice_config.yaml"
    os.makedirs("config_files", exist_ok=True)
    
    if not os.path.exists(config_path):
        # Create a default configuration
        default_config = {
            'experiment_name': 'invoice_ocr_easyocr',
            'train_data': os.getenv("TRAIN_DATA_DIR", "./ml"),
            'batch_size': 32,
            'num_iter': 1000,
            'lr': 0.001,
            'lang_char': 'None',  # Will be computed from data
            'number': '0123456789',
            'symbol': '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f)
    
    # Load configuration
    opt = get_config(config_path)
    
    # Run training
    success = train(opt, amp=False)
    
    return success


if __name__ == "__main__":
    success = run_training()
    if success:
        logger.info("Training process completed successfully!")
    else:
        logger.error("Training process failed!")
        exit(1)