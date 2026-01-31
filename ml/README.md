# Model Training Process

This directory contains scripts for improving invoice processing accuracy with corrected data from user feedback.
Using EasyOCR which doesn't require traditional model retraining - instead we use corrected data to enhance post-processing rules.

## Files:

- `train_model.py`: Main training script that processes corrections and improves validation rules

## Improvement Process:

1. User corrections are saved as cropped images in weekly directories (e.g., `25012026_01022026/`)
2. Each cropped image and its corrected text are registered in `labels.csv` with columns: `filename,words`
3. The training script analyzes these corrections to improve validation rules and field extraction accuracy
4. Updated rules are applied to enhance future invoice processing

## Usage:

```bash
# Run the improvement process
python ml/train_model.py
```

The process automatically finds the most recent weekly directory with training data and uses it to improve processing accuracy.