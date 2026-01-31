import os
import asyncio
import json
from typing import Dict, Any, List, Optional
from loguru import logger
from groq import Groq
from api.services.rag.storage_service import get_rag_context

class LLMOCRPostProcessor:
    def __init__(self, llm_provider: str = None):
        # Determine which LLM provider to use - can be passed as parameter or from environment
        if llm_provider:
            self.llm_provider = llm_provider.lower()
        else:
            self.llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()

        if self.llm_provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY", "")
            if not self.api_key:
                raise ValueError("GROQ_API_KEY environment variable is required when using Groq")
            self.client = Groq(api_key=self.api_key)
            self.model = os.getenv("GROQ_MODEL", "moonshotai/kimi-k2-instruct-0905")  # Using the Kimi Moonshot model as recommended
        else:  # Default to Groq as fallback
            self.api_key = os.getenv("GROQ_API_KEY", "")
            if not self.api_key:
                raise ValueError("GROQ_API_KEY environment variable is required")
            self.client = Groq(api_key=self.api_key)
            self.model = os.getenv("GROQ_MODEL", "moonshotai/kimi-k2-instruct-0905")  # Using the Kimi Moonshot model as recommended
        
    def process_raw_ocr(self, raw_ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw OCR result using LLM to extract structured invoice fields

        Args:
            raw_ocr_result: Dictionary containing raw OCR results with keys:
                           - 'raw_text': Combined text from OCR
                           - 'words': List of word objects with text and confidence
                           - 'image_shape': Shape of the processed image

        Returns:
            Dictionary with structured invoice fields
        """
        try:
            raw_text = raw_ocr_result.get("raw_text", "")
            ocr_words = raw_ocr_result.get("words", [])

            # Retrieve similar corrections from Qdrant for RAG
            rag_contexts = get_rag_context(raw_text, limit=5)

            # Format RAG context for the prompt
            rag_context_str = ""
            if rag_contexts:
                rag_context_str = "Similar past corrections for reference:\n"
                for i, context in enumerate(rag_contexts, 1):
                    predicted = context.get("predicted_fields", {})
                    corrected = context.get("corrected_fields", {})
                    rag_context_str += f"\nExample {i}:\n"
                    rag_context_str += f"- Predicted: {predicted}\n"
                    rag_context_str += f"- Corrected: {corrected}\n"
            else:
                rag_context_str = "No similar past corrections found."

            # Create a structured prompt for the LLM to extract fields and provide confidence
            prompt = f"""
            You are an expert invoice data extraction system. Analyze the following OCR-extracted text from an invoice and extract the following fields with confidence scores.

            Raw OCR text from the invoice:
            {raw_text}

            {rag_context_str}

            Based on the raw OCR text and the similar past corrections, extract the following fields and provide confidence scores for each:
            - vendor: Name of the company providing the goods/services
            - invoice_no: Invoice number
            - invoice_date: Invoice date in YYYY-MM-DD format
            - total: Total amount due
            - tax: Tax amount (if separately listed)
            - debit_account: Suggested accounting category based on vendor
            - credit_account: Default accounting category for credits (typically "Accounts Payable")

            Rate the confidence of each field extraction on a scale of 0.0 to 1.0, where:
            - 1.0 means the field clearly appears in the raw text with high certainty
            - 0.7-0.9 means the field appears in the raw text with good certainty
            - 0.4-0.6 means the field appears but with uncertainty or partial matches
            - 0.1-0.3 means the field has weak match to raw text or is uncertain
            - 0.0 means the field does not appear in the raw text

            Respond with a JSON object containing the extracted fields and their confidence scores in the following format:
            {{
                "extraction_results": {{
                    "vendor": "ABC Corporation",
                    "invoice_no": "INV-2026-001",
                    "invoice_date": "2026-01-15",
                    "total": 1560.00,
                    "tax": 93.60,
                    "debit_account": "General Expense",
                    "credit_account": "Accounts Payable"
                }},
                "confidence_scores": {{
                    "vendor": 0.85,
                    "invoice_no": 0.92,
                    "invoice_date": 0.78,
                    "total": 0.95,
                    "tax": 0.65,
                    "debit_account": 0.88,
                    "credit_account": 0.95
                }}
            }}

            Only respond with the JSON object, no other text.
            """

            # Call the LLM to extract structured data
            response = self._call_llm(prompt)

            # Debug logging to see the LLM response
            logger.debug(f"LLM response: {response}")

            # Parse the JSON response
            try:
                full_response = json.loads(response)

                # Extract the results and confidence scores
                extracted_data = full_response.get("extraction_results", {})
                field_confidences = full_response.get("confidence_scores", {})

                # Debug logging to see the parsed response
                logger.debug(f"LLM extraction_results: {extracted_data}")
                logger.debug(f"LLM confidence_scores: {field_confidences}")

                # Validate the extracted data has required fields
                required_fields = ["vendor", "invoice_no", "invoice_date", "total", "tax", "debit_account", "credit_account"]
                for field in required_fields:
                    if field not in extracted_data:
                        logger.debug(f"Field '{field}' not found in extraction_results, setting to None")
                        extracted_data[field] = None
                    if field not in field_confidences:
                        logger.debug(f"Field '{field}' not found in confidence_scores, setting to default 0.5")
                        field_confidences[field] = 0.5  # Default confidence if not provided

                # Add the field-specific confidences to the result
                extracted_data["field_llm_confidences"] = field_confidences

                return extracted_data
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing LLM response as JSON: {e}")
                logger.error(f"LLM response: {response}")

                # Return a default structure if JSON parsing fails
                return {
                    "vendor": None,
                    "invoice_no": None,
                    "invoice_date": None,
                    "total": None,
                    "tax": None,
                    "debit_account": "General Expense",
                    "credit_account": "Accounts Payable",
                    "confidence_details": {},
                    "field_llm_confidences": {
                        "vendor": 0.5,
                        "invoice_no": 0.5,
                        "invoice_date": 0.5,
                        "total": 0.5,
                        "tax": 0.5,
                        "debit_account": 0.5,
                        "credit_account": 0.5
                    }
                }

        except Exception as e:
            logger.error(f"Error in LLM OCR post-processing: {e}")
            # Return a default structure in case of error
            return {
                "vendor": None,
                "invoice_no": None,
                "invoice_date": None,
                "total": None,
                "tax": None,
                "debit_account": "General Expense",
                "credit_account": "Accounts Payable",
                "confidence_details": {},
                "field_llm_confidences": {
                    "vendor": 0.5,
                    "invoice_no": 0.5,
                    "invoice_date": 0.5,
                    "total": 0.5,
                    "tax": 0.5,
                    "debit_account": 0.5,
                    "credit_account": 0.5
                }
            }

    def _calculate_extraction_confidence(self, raw_text: str, ocr_words: List[Dict], extracted_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate confidence for each extracted field based on similarity with raw OCR text
        """
        confidence_details = {}

        for field_name, extracted_value in extracted_data.items():
            if field_name == "confidence_details":  # Skip the confidence details field itself
                continue

            if extracted_value is None or extracted_value == "":
                confidence_details[field_name] = 0.0
            else:
                # Calculate confidence based on how well the extracted value appears in the raw OCR text
                confidence = self._calculate_similarity_confidence(str(extracted_value), raw_text, ocr_words)
                confidence_details[field_name] = confidence

        return confidence_details

    def _calculate_similarity_confidence(self, extracted_value: str, raw_text: str, ocr_words: List[Dict]) -> float:
        """
        Calculate confidence based on similarity between extracted value and OCR results
        """
        if not extracted_value or not raw_text:
            return 0.0

        extracted_lower = extracted_value.lower().strip()
        raw_lower = raw_text.lower().strip()

        # Check if the extracted value appears in the raw text
        if extracted_lower in raw_lower:
            # If exact match in raw text, return high confidence
            return 0.9
        else:
            # Check if parts of the extracted value appear in OCR words
            matching_words = []
            extracted_parts = extracted_lower.split()

            for word_obj in ocr_words:
                word_text = word_obj.get("text", "").lower().strip()
                if word_text:
                    # Check if any part of the extracted value matches this OCR word
                    for part in extracted_parts:
                        if part in word_text or word_text in part:
                            matching_words.append(word_obj)
                            break

            if matching_words:
                # Calculate average confidence of matching words
                avg_word_conf = sum(word_obj.get("confidence", 0.5) for word_obj in matching_words) / len(matching_words)
                # Apply a slight penalty since it's not an exact match in raw text
                return min(0.9, avg_word_conf * 0.9)
            else:
                # No matches found, return low confidence
                return 0.2

    def _call_llm(self, prompt: str) -> str:
        """Make a call to the selected LLM API"""
        try:
            # Use Groq API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1000,  # Enough tokens for structured response
                top_p=1,
                stream=False,
                stop=None
            )
            return completion.choices[0].message.content
        except Exception as e:
            # Log the error and return a default response
            logger.error(f"Error calling {self.llm_provider.upper()} API: {str(e)}")
            raise