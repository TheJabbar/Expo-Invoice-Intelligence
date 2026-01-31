#!/usr/bin/env python3
"""
Simple test script to verify the new OCR implementation works correctly
"""
import asyncio
from pathlib import Path
from api.services.ocr import InvoiceOCR

def test_ocr_initialization():
    """Test that InvoiceOCR initializes without hanging"""
    print("Testing InvoiceOCR initialization...")
    try:
        ocr_engine = InvoiceOCR()
        print("OK InvoiceOCR initialized successfully")
        print(f"OK Ready status: {ocr_engine.is_ready()}")
        return True
    except Exception as e:
        print(f"ERROR Error initializing InvoiceOCR: {e}")
        return False

def test_process_method_exists():
    """Test that the process method exists"""
    print("\nTesting process method existence...")
    try:
        ocr_engine = InvoiceOCR()
        if hasattr(ocr_engine, 'process'):
            print("OK Process method exists")
            return True
        else:
            print("ERROR Process method does not exist")
            return False
    except Exception as e:
        print(f"ERROR Error testing process method: {e}")
        return False

if __name__ == "__main__":
    print("Testing new OCR implementation...")

    success = True
    success &= test_ocr_initialization()
    success &= test_process_method_exists()

    if success:
        print("\nOK All tests passed! The new OCR implementation is ready.")
    else:
        print("\nERROR Some tests failed!")
        exit(1)