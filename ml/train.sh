#!/bin/bash
# Training script for improving invoice processing with corrected data
# Using EasyOCR which doesn't require traditional model retraining
# Instead, we'll use corrected data to improve post-processing rules

set -e  # Exit on any error

echo "Starting invoice processing improvement with corrected data..."

# Find the most recent weekly training directory
ML_DIR="/app/ml"
LATEST_WEEKLY_DIR=$(ls -td ${ML_DIR}/*_* 2>/dev/null | head -n 1)

if [ -z "$LATEST_WEEKLY_DIR" ]; then
    echo "Error: No weekly training directories found"
    exit 1
fi

echo "Using weekly directory: $LATEST_WEEKLY_DIR"

# Check for labels.csv instead of train.txt
LABELS_CSV="$LATEST_WEEKLY_DIR/labels.csv"

if [ ! -f "$LABELS_CSV" ]; then
    echo "Warning: $LABELS_CSV not found, checking for legacy train.txt"

    # Check for legacy train.txt and convert if needed
    TRAIN_TXT="$LATEST_WEEKLY_DIR/train.txt"
    if [ -f "$TRAIN_TXT" ]; then
        echo "Found legacy train.txt, converting to labels.csv format..."

        # Create a simple Python script to convert train.txt to labels.csv
        python3 << EOF
import pandas as pd
import sys
from pathlib import Path

def convert_train_txt_to_csv(train_txt_path, output_csv_path):
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
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv_path, index=False)
        print(f"Converted {len(rows)} entries to {output_csv_path}")
    else:
        print("No valid entries found in train.txt for conversion")

if __name__ == "__main__":
    train_txt_path = "$TRAIN_TXT"
    output_csv_path = "$LABELS_CSV"
    convert_train_txt_to_csv(train_txt_path, output_csv_path)
EOF
    else
        echo "Error: Neither labels.csv nor train.txt found in $LATEST_WEEKLY_DIR"
        exit 1
    fi
fi

# Count the number of training samples
if command -v python3 &> /dev/null; then
    SAMPLE_COUNT=$(python3 -c "
import pandas as pd
df = pd.read_csv('$LABELS_CSV')
print(len(df))
")
    echo "Found $SAMPLE_COUNT training samples in $LABELS_CSV"
else
    echo "Warning: Python3 not available, skipping sample count"
    SAMPLE_COUNT=0
fi

# Process the corrections to improve validation rules
echo "Processing corrections to improve invoice processing accuracy..."
python3 /app/ml/train_model.py

echo "Training process completed successfully!"