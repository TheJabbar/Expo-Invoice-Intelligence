import os
import uuid
from pathlib import Path
from PIL import Image
from loguru import logger
import json
from typing import Dict, Any, Optional


def save_corrected_image(
    original_file_path: str, 
    corrected_fields: Dict[str, Any], 
    invoice_id: str,
    prediction_id: str
) -> Optional[str]:
    """
    Save the corrected image with metadata to the corrections directory
    
    Args:
        original_file_path: Path to the original uploaded image
        corrected_fields: Dictionary containing corrected field values
        invoice_id: Unique identifier for the invoice
        prediction_id: Unique identifier for the prediction
        
    Returns:
        Path to the saved corrected image file or None if failed
    """
    try:
        # Define the corrections directory
        corrections_dir = Path(os.getenv("CORRECTIONS_DIR", "/app/corrections"))
        images_dir = corrections_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the original image
        original_image = Image.open(original_file_path)
        
        # Generate a unique filename for the corrected image using invoice ID
        file_extension = Path(original_file_path).suffix
        corrected_filename = f"{invoice_id}{file_extension}"
        corrected_image_path = images_dir / corrected_filename
        
        # Save the image to the corrections directory
        original_image.save(corrected_image_path)
        
        # Create metadata file with correction information
        metadata = {
            "invoice_id": invoice_id,
            "prediction_id": prediction_id,
            "original_file": original_file_path,
            "corrected_fields": corrected_fields,
            "saved_at": str(json.dumps(corrected_fields, default=str))  # Just for reference
        }

        metadata_filename = f"{invoice_id}_metadata.json"
        metadata_path = images_dir / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved corrected image to {corrected_image_path} with metadata {metadata_path}")
        return str(corrected_image_path)
        
    except Exception as e:
        logger.error(f"Error saving corrected image: {e}")
        return None


def save_corrected_image_from_bytes(
    image_bytes: bytes, 
    corrected_fields: Dict[str, Any], 
    invoice_id: str,
    prediction_id: str,
    file_extension: str = ".png"
) -> Optional[str]:
    """
    Save the corrected image from bytes data with metadata to the corrections directory
    
    Args:
        image_bytes: Bytes of the image to save
        corrected_fields: Dictionary containing corrected field values
        invoice_id: Unique identifier for the invoice
        prediction_id: Unique identifier for the prediction
        file_extension: Extension for the saved image file
        
    Returns:
        Path to the saved corrected image file or None if failed
    """
    try:
        # Define the corrections directory
        corrections_dir = Path(os.getenv("CORRECTIONS_DIR", "/app/corrections"))
        images_dir = corrections_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the image from bytes
        from io import BytesIO
        image_stream = BytesIO(image_bytes)
        image = Image.open(image_stream)
        
        # Generate a unique filename for the corrected image using invoice ID
        corrected_filename = f"{invoice_id}{file_extension}"
        corrected_image_path = images_dir / corrected_filename
        
        # Save the image to the corrections directory
        image.save(corrected_image_path)
        
        # Create metadata file with correction information
        metadata = {
            "invoice_id": invoice_id,
            "prediction_id": prediction_id,
            "corrected_fields": corrected_fields,
            "file_size": len(image_bytes),
            "saved_at": str(json.dumps(corrected_fields, default=str))  # Just for reference
        }

        metadata_filename = f"{invoice_id}_metadata.json"
        metadata_path = images_dir / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved corrected image from bytes to {corrected_image_path} with metadata {metadata_path}")
        return str(corrected_image_path)
        
    except Exception as e:
        logger.error(f"Error saving corrected image from bytes: {e}")
        return None