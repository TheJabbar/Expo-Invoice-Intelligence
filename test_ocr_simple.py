#!/usr/bin/env python3
"""
Create a simple test image and test the OCR functionality
"""
import cv2
import numpy as np
from pathlib import Path
import tempfile

# Create a simple test image with some text
def create_test_image():
    # Create a white image
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    # Add some text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "INVOICE SAMPLE TEXT 12345"
    cv2.putText(img, text, (50, 100), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    cv2.imwrite(temp_file.name, img)
    return Path(temp_file.name)

def test_ocr_with_sample_image():
    """Test OCR with a sample image"""
    print("Creating sample image for OCR test...")
    image_path = create_test_image()
    
    try:
        print(f"Testing OCR on image: {image_path}")
        from api.services.ocr import InvoiceOCR
        
        # Initialize OCR engine (this will download models if needed)
        print("Initializing OCR engine...")
        ocr_engine = InvoiceOCR()
        
        print("Processing image...")
        result = ocr_engine.process(image_path)
        
        print(f"OCR Result:")
        print(f"  Raw text: {result['raw_text']}")
        print(f"  Number of words detected: {len(result['words'])}")
        print(f"  Image shape: {result['image_shape']}")
        
        # Clean up
        image_path.unlink()
        
        # Check if we got reasonable results
        if result['raw_text'] and len(result['words']) > 0:
            print("SUCCESS: OCR worked correctly!")
            return True
        else:
            print("WARNING: OCR returned empty results")
            return False
            
    except Exception as e:
        print(f"ERROR during OCR test: {e}")
        # Clean up even if there's an error
        try:
            image_path.unlink()
        except:
            pass
        return False

if __name__ == "__main__":
    print("Testing OCR with sample image...")
    success = test_ocr_with_sample_image()
    
    if success:
        print("\nOCR test completed successfully!")
    else:
        print("\nOCR test failed!")
        exit(1)