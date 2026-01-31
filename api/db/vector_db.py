import json
from typing import Dict, List, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import os
from loguru import logger

# Initialize sentence transformer model for embeddings
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Initialize Qdrant client with retry mechanism
def get_qdrant_client():
    try:
        # Try to connect to Qdrant service
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Test connection
        client.health()
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return client
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}, using in-memory mode: {e}")
        return QdrantClient(":memory:")

qdrant_client = get_qdrant_client()


def initialize_collections():
    """Initialize Qdrant collections if they don't exist"""
    try:
        # Check if collection exists
        qdrant_client.get_collection(collection_name="invoice_corrections")
    except:
        # Create collection if it doesn't exist
        qdrant_client.create_collection(
            collection_name="invoice_corrections",
            vectors_config=models.VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=models.Distance.COSINE)
        )
        logger.info("Created Qdrant collection 'invoice_corrections'")


def store_correction_for_rag(ocr_text: str, predicted_fields: Dict, corrected_fields: Dict):
    """Store correction data in Qdrant for RAG"""
    import uuid
    try:
        # Combine OCR text and predicted fields to create a searchable context
        context_text = f"OCR: {ocr_text}. Predicted: {json.dumps(predicted_fields)}. Corrected: {json.dumps(corrected_fields)}"

        # Generate embedding for the context
        embedding = encoder.encode(context_text).tolist()

        # Prepare payload
        payload = {
            "ocr_text": ocr_text,
            "predicted_fields": predicted_fields,
            "corrected_fields": corrected_fields,
            "created_at": str(datetime.utcnow())
        }

        # Store in Qdrant
        qdrant_client.upsert(
            collection_name="invoice_corrections",
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID
                    vector=embedding,
                    payload=payload
                )
            ]
        )

        logger.info(f"Stored correction in Qdrant for RAG")
    except Exception as e:
        logger.error(f"Error storing correction in Qdrant: {e}")


def get_similar_corrections(query_text: str, limit: int = 5) -> List[Dict]:
    """Retrieve similar corrections from Qdrant for RAG"""
    try:
        # Generate embedding for the query
        query_embedding = encoder.encode(query_text).tolist()

        # Query for similar contexts using query_points (newer Qdrant API)
        search_results = qdrant_client.query_points(
            collection_name="invoice_corrections",
            query=query_embedding,
            limit=limit
        )

        # Extract payloads from results
        # In the newer API, results are accessed differently
        similar_corrections = [hit.payload for hit in search_results.points]

        logger.info(f"Retrieved {len(similar_corrections)} similar corrections from Qdrant")
        return similar_corrections
    except Exception as e:
        logger.error(f"Error retrieving similar corrections from Qdrant: {e}")
        return []


# Initialize collections on module load
initialize_collections()