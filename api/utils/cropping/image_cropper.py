import os
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime, timedelta


def crop_corrected_regions(
    original_image_path: str,
    ocr_result: Dict[str, Any],
    corrected_fields: Dict[str, Any],
    invoice_id: str
) -> List[str]:
    """
    Crop regions of the image that correspond to corrected fields and save them as training data

    Args:
        original_image_path: Path to the original image
        ocr_result: OCR result containing raw text and word bounding boxes
        corrected_fields: Dictionary containing corrected field values
        invoice_id: Invoice ID for naming the cropped images

    Returns:
        List of paths to the saved cropped images
    """
    try:
        # Load the original image
        image = cv2.imread(original_image_path)
        if image is None:
            logger.error(f"Could not load image: {original_image_path}")
            return []

        # Get the words from OCR result
        ocr_words = ocr_result.get("words", [])
        saved_paths = []

        # Create weekly timestamp directory
        current_date = datetime.now()
        # Format: DDMMYYYY_DDMMYYYY (e.g., 25012026_01022026) - using current date for both parts as per example
        day_month_year = current_date.strftime("%d%m%Y")
        timestamp = f"{day_month_year}_{day_month_year}"
        weekly_train_dir = Path(os.getenv("TRAIN_DATA_DIR", "/app/ml")) / timestamp
        images_dir = weekly_train_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Collect all crops to write to CSV at the end
        csv_rows = []

        # For each corrected field, find the corresponding OCR text and crop the region
        for field_name, corrected_value in corrected_fields.items():
            if field_name in ["vendor", "invoice_no", "invoice_date", "total", "tax", "debit_account", "credit_account"]:
                # Find the OCR text that corresponds to this field
                matched_indices = find_matching_ocr_indices(ocr_words, corrected_value)

                for idx in matched_indices:
                    if idx < len(ocr_words):
                        word_info = ocr_words[idx]
                        bbox = word_info.get("bbox", [])

                        if bbox and len(bbox) >= 4:
                            # Convert bbox to pixel coordinates
                            # bbox format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]

                            x_min, x_max = int(min(x_coords)), int(max(x_coords))
                            y_min, y_max = int(min(y_coords)), int(max(y_coords))

                            # Add padding to the crop
                            padding = 5
                            x_min = max(0, x_min - padding)
                            x_max = min(image.shape[1], x_max + padding)
                            y_min = max(0, y_min - padding)
                            y_max = min(image.shape[0], y_max + padding)

                            # Crop the region
                            cropped_region = image[y_min:y_max, x_min:x_max]

                            if cropped_region.size > 0:
                                # Generate filename with field name and sequence number
                                field_sequence = len([p for p in saved_paths if field_name in os.path.basename(p)])
                                cropped_filename = f"{invoice_id}_{field_name}_{field_sequence}.png"
                                cropped_path = images_dir / cropped_filename

                                # Save the cropped region
                                cv2.imwrite(str(cropped_path), cropped_region)
                                saved_paths.append(str(cropped_path))

                                # Add to CSV rows for later writing
                                relative_path = str(cropped_path.relative_to(weekly_train_dir))
                                csv_rows.append({
                                    'filename': relative_path,
                                    'words': str(corrected_value)
                                })

                                # Also save metadata about the crop
                                metadata = {
                                    "original_image": str(original_image_path),
                                    "invoice_id": invoice_id,
                                    "field_name": field_name,
                                    "corrected_value": str(corrected_value),
                                    "bbox": bbox,
                                    "crop_coordinates": [x_min, y_max, x_max, y_max],
                                    "original_text": word_info.get("text", ""),
                                    "confidence": word_info.get("confidence", 0.0)
                                }

                                metadata_path = images_dir / f"{invoice_id}_{field_name}_{field_sequence}_metadata.json"
                                with open(metadata_path, 'w') as f:
                                    json.dump(metadata, f, indent=2)

                                logger.info(f"Cropped and saved region for {field_name} in {invoice_id}: {cropped_path}")

        # Write all collected data to labels.csv
        if csv_rows:
            labels_csv_path = weekly_train_dir / "labels.csv"
            import pandas as pd

            # Check if CSV already exists to append to it
            if labels_csv_path.exists():
                existing_df = pd.read_csv(labels_csv_path)
                new_df = pd.DataFrame(csv_rows)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = pd.DataFrame(csv_rows)

            combined_df.to_csv(labels_csv_path, index=False)
            logger.info(f"Saved {len(csv_rows)} entries to {labels_csv_path}")

        return saved_paths

    except Exception as e:
        logger.error(f"Error cropping corrected regions: {e}")
        return []


def find_matching_ocr_indices(ocr_words: List[Dict], corrected_value: Any) -> List[int]:
    """
    Find indices of OCR words that match the corrected value
    
    Args:
        ocr_words: List of OCR word dictionaries
        corrected_value: The corrected value to match
        
    Returns:
        List of indices of matching OCR words
    """
    matches = []
    corrected_str = str(corrected_value).lower().strip()
    
    for i, word_info in enumerate(ocr_words):
        text = word_info.get("text", "").lower().strip()
        
        # Check if the OCR text contains or matches the corrected value
        if corrected_str in text or text in corrected_str:
            matches.append(i)
        # Additional fuzzy matching could be added here if needed
    
    return matches