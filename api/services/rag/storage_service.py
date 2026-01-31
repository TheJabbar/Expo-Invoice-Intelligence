from api.db.vector_db import store_correction_for_rag, get_similar_corrections
from loguru import logger
import json
from typing import Dict, Any


def store_corrected_prediction(ocr_text: str, predicted_fields: Dict, corrected_fields: Dict):
    """Store corrected predictions in Qdrant for RAG"""
    try:
        store_correction_for_rag(ocr_text, predicted_fields, corrected_fields)
        logger.info(f"Stored corrected prediction in Qdrant for RAG")
    except Exception as e:
        logger.error(f"Error storing corrected prediction in Qdrant: {e}")
        raise


def get_rag_context(query_text: str, limit: int = 5) -> list:
    """Get similar corrections from Qdrant for RAG"""
    try:
        similar_corrections = get_similar_corrections(query_text, limit)
        logger.info(f"Retrieved {len(similar_corrections)} similar corrections for RAG")
        return similar_corrections
    except Exception as e:
        logger.error(f"Error retrieving RAG context: {e}")
        return []