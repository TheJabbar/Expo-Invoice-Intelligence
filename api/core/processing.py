from pathlib import Path
import uuid
import json
from loguru import logger
from api.services.ocr import InvoiceOCR
from api.services.extractor import FieldExtractor
from api.core.db import save_prediction, save_correction
from api.services.rag.storage_service import store_corrected_prediction
from api.utils.image_storage import save_corrected_image
from api.utils.cropping.image_cropper import crop_corrected_regions


class InvoiceProcessor:
    def __init__(self):
        self.ocr_engine = InvoiceOCR()
        self.extractor = FieldExtractor()

    async def process_invoice(self, file, auto_post: bool = False):
        """Process an uploaded invoice file and extract fields"""
        logger.info(f"Received invoice upload request for file: {file.filename}")

        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
            logger.warning(f"Invalid file type for: {file.filename}")
            raise ValueError("Only PDF, JPG, PNG files allowed")

        # Generate unique invoice ID
        invoice_id = f"INV-{uuid.uuid4().hex[:8]}"
        file_ext = Path(file.filename).suffix.lower()
        # Use a more persistent location that's accessible to the cropping function
        file_path = Path(f"/app/temp/{invoice_id}{file_ext}")
        # Create the temp directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Saving file to temporary path: {file_path}")

        # Save uploaded file
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        except Exception as e:
            logger.error(f"File save failed: {str(e)}")
            raise ValueError(f"File save failed: {str(e)}")

        # OCR processing
        try:
            logger.info(f"Starting OCR processing for invoice: {invoice_id}")
            ocr_result = self.ocr_engine.process(file_path)
            logger.debug(f"OCR result for invoice {invoice_id}: {ocr_result}")
            logger.info(f"OCR processing completed for invoice: {invoice_id}")
        except Exception as e:
            logger.error(f"OCR failed for invoice {invoice_id}: {str(e)}")
            raise ValueError(f"OCR failed: {str(e)}")
        finally:
            # DON'T delete the file yet - we need it for cropping when corrections come in
            # file_path.unlink(missing_ok=True)  # Cleanup temp file
            logger.debug(f"File kept for potential cropping: {file_path}")

        # Field extraction
        logger.info(f"Starting field extraction for invoice: {invoice_id}")
        extraction = self.extractor.extract(ocr_result)

        # Confidence gating - check if any field confidence is below threshold
        field_confidences = extraction["field_confidences"]
        requires_review = any(conf < 0.75 for conf in field_confidences.values())

        # Save prediction to DB
        logger.info(f"Saving prediction for invoice: {invoice_id}")
        prediction_id = save_prediction(
            invoice_id=invoice_id,
            model_version="v1.0-ppocrv4",
            predicted_fields=extraction["fields"],
            confidence=extraction["confidence"],
            field_confidences=extraction["field_confidences"],
            raw_ocr=json.dumps(ocr_result),
            original_file_path=str(file_path)  # Pass the original file path for cropping
        )
        logger.info(f"Prediction saved with ID: {prediction_id}")

        # Prepare the response in the desired format
        fields = extraction["fields"]
        field_confidences = extraction["field_confidences"]

        response = {
            "invoice_id": invoice_id,
            "prediction_id": prediction_id,  # Include prediction_id for feedback
            "vendor": fields.get("vendor"),
            "invoice_no": fields.get("invoice_no"),
            "invoice_date": fields.get("invoice_date"),
            "tax": fields.get("tax"),
            "total": fields.get("total"),
            "debit_account": fields.get("debit_account"),
            "credit_account": fields.get("credit_account") or "Accounts Payable",
            "confidence": round(extraction["confidence"], 2),
            "model_version": "v1.0-ppocr",
            "field_confidences": {k: round(v, 2) for k, v in field_confidences.items()}
        }
        logger.debug(f"Response for invoice {invoice_id}: {response}")

        # No longer storing passed predictions in SQLite - only corrections are stored in JSON

        # Auto-posting safety check
        if auto_post and not requires_review:
            response["journal_entry"] = {
                "debit_account": extraction["fields"]["debit_account"],
                "credit_account": "Accounts Payable",
                "amount": extraction["fields"]["total"],
                "memo": f"Invoice {extraction['fields']['invoice_no']} from {extraction['fields']['vendor']}"
            }

        logger.info(f"Invoice processing completed for ID: {invoice_id}")
        return response

    def submit_feedback(self, prediction_id: str, corrected_fields: str, user_id: str = "anonymous", image_file=None):
        """Submit feedback for a prediction (kept for backward compatibility)"""
        logger.info(f"Received feedback for prediction ID: {prediction_id}")

        try:
            corrections = json.loads(corrected_fields)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in corrected_fields for prediction ID: {prediction_id}")
            raise ValueError("Invalid JSON in corrected_fields")

        if not corrections:
            logger.warning(f"No corrections provided for prediction ID: {prediction_id}")
            raise ValueError("No corrections provided")

        # Find the original prediction record to get the raw OCR
        from api.core.db import _load_data, PREDICTIONS_FILE
        predictions = _load_data(PREDICTIONS_FILE)

        prediction_record = None
        for pred in predictions:
            if pred["id"] == prediction_id:
                prediction_record = pred
                break

        if not prediction_record:
            logger.error(f"No prediction found for ID: {prediction_id}")
            raise ValueError(f"No prediction found for prediction ID: {prediction_id}")

        # Save correction (with safety validation later in training pipeline)
        logger.info(f"Saving correction for prediction ID: {prediction_id}")
        correction_id = save_correction(
            prediction_id=prediction_id,
            corrected_fields=corrections,
            user_id=user_id
        )
        logger.info(f"Correction saved with ID: {correction_id}")

        # Store in Qdrant for RAG
        try:
            raw_ocr = prediction_record.get("raw_ocr", "")
            predicted_fields = prediction_record.get("predicted_fields", {})
            store_corrected_prediction(raw_ocr, predicted_fields, corrections)
        except Exception as e:
            logger.error(f"Error storing correction in Qdrant: {e}")

        # Crop and save corrected regions as training data
        try:
            # Get the original file path from the prediction record
            original_file_path = prediction_record.get("original_file_path")
            if original_file_path:
                # Load the OCR result from the prediction record
                ocr_result = json.loads(prediction_record.get("raw_ocr", "{}"))

                # Get the invoice ID from the prediction record
                invoice_id = prediction_record.get("invoice_id", "")

                # Crop and save the corrected regions
                cropped_paths = crop_corrected_regions(
                    original_image_path=original_file_path,
                    ocr_result=ocr_result,
                    corrected_fields=corrections,
                    invoice_id=invoice_id
                )

                logger.info(f"Cropped {len(cropped_paths)} regions for training data for invoice {invoice_id}")

                # Clean up the original temporary file after cropping
                import os
                if os.path.exists(original_file_path):
                    os.remove(original_file_path)
                    logger.debug(f"Cleaned up temporary file: {original_file_path}")
        except Exception as e:
            logger.error(f"Error cropping corrected regions: {e}")

        # Save corrected image if provided
        try:
            if image_file:
                # Save the uploaded image file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=image_file.filename) as temp_file:
                    content = image_file.file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                # Reset file pointer for possible reuse
                image_file.file.seek(0)

                # Save the corrected image with metadata
                save_corrected_image(
                    original_file_path=temp_file_path,
                    corrected_fields=corrections,
                    invoice_id=prediction_record.get("invoice_id", ""),
                    prediction_id=prediction_id
                )

                # Clean up temporary file
                import os
                os.unlink(temp_file_path)
        except Exception as e:
            logger.error(f"Error saving corrected image: {e}")

        return {
            "feedback_id": correction_id,
            "status": "recorded",
            "training_eligible_after": "Model learns and rebuilds every week"
        }

    def submit_feedback_by_invoice_id(self, invoice_id: str, corrected_fields: str, user_id: str = "anonymous", image_file=None):
        """Submit feedback for a prediction using invoice ID"""
        logger.info(f"Received feedback for invoice ID: {invoice_id}")

        try:
            corrections = json.loads(corrected_fields)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in corrected_fields for invoice ID: {invoice_id}")
            raise ValueError("Invalid JSON in corrected_fields")

        if not corrections:
            logger.warning(f"No corrections provided for invoice ID: {invoice_id}")
            raise ValueError("No corrections provided")

        # Find the prediction record by invoice_id to get the prediction_id
        from api.core.db import _load_data, PREDICTIONS_FILE
        predictions = _load_data(PREDICTIONS_FILE)

        prediction_record = None
        for pred in predictions:
            if pred["invoice_id"] == invoice_id:
                prediction_record = pred
                break

        if not prediction_record:
            logger.error(f"No prediction found for invoice ID: {invoice_id}")
            raise ValueError(f"No prediction found for invoice ID: {invoice_id}")

        prediction_id = prediction_record["id"]

        # Save correction (with safety validation later in training pipeline)
        logger.info(f"Saving correction for prediction ID: {prediction_id} (from invoice ID: {invoice_id})")
        correction_id = save_correction(
            prediction_id=prediction_id,
            corrected_fields=corrections,
            user_id=user_id
        )
        logger.info(f"Correction saved with ID: {correction_id}")

        # Store in Qdrant for RAG
        try:
            raw_ocr = prediction_record.get("raw_ocr", "")
            predicted_fields = prediction_record.get("predicted_fields", {})
            store_corrected_prediction(raw_ocr, predicted_fields, corrections)
        except Exception as e:
            logger.error(f"Error storing correction in Qdrant: {e}")

        # Crop and save corrected regions as training data
        try:
            # Get the original file path from the prediction record
            original_file_path = prediction_record.get("original_file_path")
            if original_file_path:
                # Load the OCR result from the prediction record
                ocr_result = json.loads(prediction_record.get("raw_ocr", "{}"))

                # Crop and save the corrected regions
                cropped_paths = crop_corrected_regions(
                    original_image_path=original_file_path,
                    ocr_result=ocr_result,
                    corrected_fields=corrections,
                    invoice_id=invoice_id
                )

                logger.info(f"Cropped {len(cropped_paths)} regions for training data for invoice {invoice_id}")

                # Clean up the original temporary file after cropping
                import os
                if os.path.exists(original_file_path):
                    os.remove(original_file_path)
                    logger.debug(f"Cleaned up temporary file: {original_file_path}")
        except Exception as e:
            logger.error(f"Error cropping corrected regions: {e}")

        # Save corrected image if provided
        try:
            if image_file:
                # Save the uploaded image file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=image_file.filename) as temp_file:
                    content = image_file.file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                # Reset file pointer for possible reuse
                image_file.file.seek(0)

                # Save the corrected image with metadata
                save_corrected_image(
                    original_file_path=temp_file_path,
                    corrected_fields=corrections,
                    invoice_id=invoice_id,
                    prediction_id=prediction_id
                )

                # Clean up temporary file
                import os
                os.unlink(temp_file_path)
        except Exception as e:
            logger.error(f"Error saving corrected image: {e}")

        return {
            "feedback_id": correction_id,
            "status": "recorded",
            "training_eligible_after": "Model learns and rebuilds every week"
        }