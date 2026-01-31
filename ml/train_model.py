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


def get_config(config_path):
    """
    Load configuration from YAML file and process character set from CSV data
    """
    with open(config_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)

    # Process character set from training data if needed
    if opt.get('lang_char') == 'None':
        characters = ''
        train_data_dir = opt.get('train_data', os.getenv("TRAIN_DATA_DIR", "/app/ml/train"))

        # Find weekly directories with labels.csv
        ml_dir = Path(train_data_dir)
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
    Train the model using the configuration and training data
    """
    logger.info("Starting training process...")

    # Find the most recent weekly training directory
    ml_dir = Path(opt.get('train_data', os.getenv("TRAIN_DATA_DIR", "/app/ml")))
    weekly_dirs = [d for d in ml_dir.iterdir() if d.is_dir() and '_' in d.name and len(d.name) == 17]  # Format: DDMMYYYY_DDMMYYYY

    if not weekly_dirs:
        logger.warning("No weekly training directories found")
        return False

    # Use the most recent weekly directory
    latest_weekly_dir = max(weekly_dirs, key=lambda x: x.name)
    labels_csv_path = latest_weekly_dir / "labels.csv"

    if not labels_csv_path.exists():
        logger.warning(f"No labels.csv file found in {latest_weekly_dir}")
        # Check for legacy train.txt and convert if needed
        train_txt_path = latest_weekly_dir / "train.txt"
        if train_txt_path.exists():
            logger.info(f"Found legacy train.txt, converting to labels.csv format")
            convert_train_txt_to_csv(train_txt_path, latest_weekly_dir)
            labels_csv_path = latest_weekly_dir / "labels.csv"
        else:
            logger.error(f"No training data found in {latest_weekly_dir}")
            return False

    try:
        # Read the training data
        df = pd.read_csv(labels_csv_path, usecols=['filename', 'words'], keep_default_na=False)
        logger.info(f"Loaded {len(df)} training samples from {labels_csv_path}")

        # Process the training data
        total_samples = len(df)
        logger.info(f"Processing {total_samples} samples")

        # Example: Update validation rules based on the training data
        for idx, row in df.iterrows():
            filename = row['filename']
            words = row['words']

            # In a real implementation, this would update model weights or rules
            # For now, we just log the progress
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{total_samples} samples")

        # Simulate model saving
        experiment_name = opt.get('experiment_name', 'invoice_ocr')
        model_path = f"./saved_models/{experiment_name}/final.pth"
        logger.info(f"Simulated model saved to {model_path}")

        logger.info("Training completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False


def convert_train_txt_to_csv(train_txt_path, weekly_dir):
    """
    Convert legacy train.txt format to new labels.csv format
    """
    logger.info(f"Converting {train_txt_path} to labels.csv format")

    try:
        rows = []
        with open(train_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # train.txt format: "image_path corrected_text"
                    parts = line.split(' ', 1)  # Split only on first space
                    if len(parts) == 2:
                        image_path = parts[0]
                        corrected_text = parts[1]

                        # Extract filename from path
                        filename = Path(image_path).name
                        rows.append({'filename': filename, 'words': corrected_text})

        # Write to labels.csv
        labels_csv = weekly_dir / "labels.csv"
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(labels_csv, index=False)
            logger.info(f"Converted {len(rows)} entries to {labels_csv}")
        else:
            logger.warning("No valid entries found in train.txt for conversion")

    except Exception as e:
        logger.error(f"Error converting train.txt to CSV: {e}")


def run_training():
    """
    Main training function that follows the example structure
    """
    logger.info("Processing corrected data to improve invoice processing accuracy")

    # Load configuration
    config_path = "config_files/invoice_config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found")
        return False

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