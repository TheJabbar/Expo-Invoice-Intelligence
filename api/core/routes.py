from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from api.core.processing import InvoiceProcessor
from api.core.db import save_correction
from api.models import (
    HealthCheck,
    InvoiceUploadResponse,
    SubmitFeedbackRequest,
    SubmitFeedbackResponse
)

router = APIRouter()
processor = InvoiceProcessor()


@router.get("/health", response_model=HealthCheck)
async def health():
    logger.info("Health check endpoint called")
    is_ready = processor.ocr_engine.is_ready()
    logger.debug(f"OCR engine ready status: {is_ready}")
    return HealthCheck(status="ok", ocr_ready=is_ready)


@router.post("/api/invoice/upload", response_model=InvoiceUploadResponse)
async def upload_invoice(
    file: UploadFile = File(...),
    auto_post: bool = Form(False)
):
    try:
        result = await processor.process_invoice(file, auto_post)
        return result
    except ValueError as e:
        # Convert ValueError to HTTPException
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in upload_invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/api/invoice/feedback", response_model=SubmitFeedbackResponse)
async def submit_feedback(
    invoice_id: str = Form(...),
    corrected_fields: str = Form(...),  # JSON string
    user_id: str = Form("anonymous"),
    image_file: UploadFile = File(None)  # Optional image file for correction
):
    try:
        result = processor.submit_feedback_by_invoice_id(invoice_id, corrected_fields, user_id, image_file)
        return result
    except ValueError as e:
        # Convert ValueError to HTTPException
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in submit_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")