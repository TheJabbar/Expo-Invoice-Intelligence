import os
from pathlib import Path
from loguru import logger


def download_base_model():
    """
    Placeholder function - EasyOCR doesn't require manual model downloads
    Models are handled automatically by the EasyOCR library
    """
    logger.info("EasyOCR handles model downloads automatically, no manual download needed")


def get_model_path():
    """
    Get the path to the base model directory
    """
    model_dir = Path(os.getenv("MODEL_DIR", "/app/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)

    return str(model_dir)


if __name__ == "__main__":
    # EasyOCR handles model downloads automatically
    logger.info("EasyOCR handles model downloads automatically, no manual download needed")