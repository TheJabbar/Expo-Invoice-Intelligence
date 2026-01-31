import json
import os
import uuid
from datetime import datetime
from pathlib import Path

# Directory to store correction data
CORRECTIONS_DIR = Path(os.getenv("CORRECTIONS_DIR", "/app/corrections"))
CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Files to store data
PREDICTIONS_FILE = CORRECTIONS_DIR / "predictions.json"
CORRECTIONS_FILE = CORRECTIONS_DIR / "corrections.json"

def _load_data(filename):
    """Load data from JSON file"""
    if filename.exists():
        with open(filename, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def _save_data(filename, data):
    """Save data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def save_prediction(
    invoice_id: str,
    model_version: str,
    predicted_fields: dict,
    confidence: float,
    field_confidences: dict,
    raw_ocr: str,
    original_file_path: str = None
) -> str:
    """Save prediction to local file storage"""
    predictions = _load_data(PREDICTIONS_FILE)

    prediction_id = str(uuid.uuid4())
    prediction_record = {
        "id": prediction_id,
        "invoice_id": invoice_id,
        "model_version": model_version,
        "predicted_fields": predicted_fields,
        "field_confidences": field_confidences,
        "overall_confidence": confidence,
        "raw_ocr": raw_ocr,
        "original_file_path": original_file_path,  # Store original file path for cropping
        "timestamp": datetime.now().isoformat()
    }

    predictions.append(prediction_record)
    _save_data(PREDICTIONS_FILE, predictions)

    return prediction_id

def save_correction(
    prediction_id: str,
    corrected_fields: dict,
    user_id: str
) -> str:
    """Save correction to local file storage"""
    corrections = _load_data(CORRECTIONS_FILE)
    predictions = _load_data(PREDICTIONS_FILE)

    # Find the original prediction record
    original_prediction = None
    for pred in predictions:
        if pred["id"] == prediction_id:
            original_prediction = pred
            break

    if not original_prediction:
        raise ValueError(f"Original prediction with ID {prediction_id} not found")

    correction_id = str(uuid.uuid4())

    # Create correction record in the expected format
    correction_record = {
        "id": correction_id,
        "invoice_id": original_prediction["invoice_id"],
        "ocr_text": original_prediction["raw_ocr"],
        "predicted_fields": original_prediction["predicted_fields"],
        "corrected_fields": corrected_fields,
        "model_version": original_prediction["model_version"],
        "timestamp": datetime.now().isoformat()
    }

    corrections.append(correction_record)
    _save_data(CORRECTIONS_FILE, corrections)

    return correction_id