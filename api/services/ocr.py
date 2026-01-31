import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from loguru import logger
import easyocr

class InvoiceOCR:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize EasyOCR reader which is more reliable and faster
            cls._instance.reader = easyocr.Reader(['en'], gpu=False)  # Use CPU to avoid GPU issues
            cls._instance._ready = True
        return cls._instance

    def is_ready(self) -> bool:
        return getattr(self, '_ready', False)

    def process(self, file_path: Path) -> dict:
        # Load image from file path
        if file_path.suffix.lower() == '.pdf':
            image = self._pdf_to_image(file_path)
        else:
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Failed to read image: {file_path}")

        # Convert image to RGB (EasyOCR expects RGB, not BGR)
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image  # Grayscale image

        try:
            # Perform OCR using EasyOCR
            # This returns a list of [bbox, text, confidence] for each detected text
            result = self.reader.readtext(image_rgb)

            # Process the results into the expected format
            words = []
            for detection in result:
                if len(detection) == 3:
                    bbox, text, confidence = detection
                else:
                    # Handle unexpected format
                    continue

                # Ensure text is a string and confidence is a float
                text_str = str(text) if text is not None else ""

                # Skip empty text results
                if not text_str.strip():
                    continue

                conf_float = float(confidence) if confidence is not None else 0.0

                # Normalize bbox format to ensure it has 4 coordinate pairs
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    normalized_bbox = []
                    for coord_pair in bbox:
                        if isinstance(coord_pair, (list, tuple)) and len(coord_pair) >= 2:
                            normalized_bbox.append([float(coord_pair[0]), float(coord_pair[1])])
                        else:
                            # Fallback if coordinate format is unexpected
                            normalized_bbox.append([0.0, 0.0])

                    # Ensure we have exactly 4 points
                    while len(normalized_bbox) < 4:
                        normalized_bbox.append([0.0, 0.0])
                    normalized_bbox = normalized_bbox[:4]
                else:
                    # Fallback bbox if the original bbox is not in expected format
                    normalized_bbox = [[0, 0], [10, 0], [10, 10], [0, 10]]

                words.append({
                    "text": text_str,
                    "confidence": conf_float,
                    "bbox": normalized_bbox
                })

            # Combine all text for raw_text
            raw_text = " ".join([w["text"] for w in words]) if words else ""

            # Log results for debugging
            logger.debug(f"OCR completed - Raw text: '{raw_text[:100]}...', Words count: {len(words)}, Image shape: {image.shape}")

            return {
                "raw_text": raw_text,
                "words": words,
                "image_shape": image.shape.tolist() if hasattr(image.shape, 'tolist') else image.shape
            }

        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            # Return empty result in case of error
            return {
                "raw_text": "",
                "words": [],
                "image_shape": image.shape.tolist() if hasattr(image.shape, 'tolist') else image.shape
            }

    def _pdf_to_image(self, pdf_path: Path) -> np.ndarray:
        try:
            images = convert_from_path(
                pdf_path,
                dpi=200,
                first_page=1,
                last_page=1,
                thread_count=1
            )
            # Convert PIL image to OpenCV format
            return cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"PDF conversion failed: {str(e)}")
            raise ValueError(f"PDF conversion failed: {str(e)}")