import math
import re
from typing import Dict, List
from loguru import logger


class ConfidenceCalculator:
    """
    Calculates confidence scores for OCR-extracted invoice fields using multiple factors:
    - OCR word confidence scores
    - Pattern matching strength
    - Field completeness
    - Cross-field consistency
    """

    def __init__(self):
        self.confidence_weights = {
            "ocr_confidence": 0.4,      # Weight for OCR word confidence
            "pattern_strength": 0.3,    # Weight for pattern matching confidence
            "completeness": 0.2,        # Weight for field completeness
            "consistency": 0.1          # Weight for cross-field consistency
        }

    def calculate_field_confidence(
        self,
        ocr_words: List[Dict],
        field_text: str,
        pattern_matched: bool = True,
        expected_format: str = None
    ) -> float:
        """
        Calculate confidence for a specific field
        
        Args:
            ocr_words: List of OCR words with confidence scores
            field_text: The extracted field text
            pattern_matched: Whether the field was extracted using a regex pattern
            expected_format: Expected format for validation (e.g., date, currency)
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not field_text or not field_text.strip():
            return 0.0

        # OCR confidence component
        ocr_conf = self._calculate_ocr_confidence(ocr_words, field_text)

        # Pattern strength component
        pattern_conf = 1.0 if pattern_matched else 0.3

        # Completeness component
        completeness_conf = self._calculate_completeness(field_text, expected_format)

        # Consistency component (placeholder - would check against other fields)
        consistency_conf = 1.0  # Simplified for this implementation

        # Weighted average
        total_conf = (
            ocr_conf * self.confidence_weights["ocr_confidence"] +
            pattern_conf * self.confidence_weights["pattern_strength"] +
            completeness_conf * self.confidence_weights["completeness"] +
            consistency_conf * self.confidence_weights["consistency"]
        )

        return min(1.0, max(0.0, total_conf))

    def calculate_overall_confidence(self, field_confidences: Dict[str, float]) -> float:
        """
        Calculate overall confidence for the entire invoice based on individual field confidences
        
        Args:
            field_confidences: Dictionary of field names to confidence scores
            
        Returns:
            Overall confidence score between 0.0 and 1.0
        """
        if not field_confidences:
            return 0.0

        # Use harmonic mean to penalize low-confidence fields more heavily
        conf_values = [conf for conf in field_confidences.values() if conf is not None]
        
        if not conf_values:
            return 0.0

        # Harmonic mean formula: n / (sum(1/x_i))
        harmonic_sum = sum(1.0 / max(conf, 0.01) for conf in conf_values)  # Avoid division by zero
        harmonic_mean = len(conf_values) / harmonic_sum

        # Adjust based on number of missing fields
        expected_fields = 6  # vendor, invoice_no, date, tax, total, debit_account
        present_fields = len(conf_values)
        completeness_factor = present_fields / expected_fields

        # Combine harmonic mean with completeness factor
        overall_conf = harmonic_mean * completeness_factor

        # Ensure we don't exceed 0.99 (reserve 0.99-1.0 for perfect cases)
        return min(0.99, overall_conf)

    def _calculate_ocr_confidence(self, ocr_words: List[Dict], field_text: str) -> float:
        """Calculate confidence based on OCR word confidence scores"""
        if not ocr_words or not field_text:
            return 0.3  # Default low confidence

        # Find words that match the field text
        field_words = []
        field_lower = field_text.lower().strip()

        # First, try exact substring matches
        for word_obj in ocr_words:
            word_text = word_obj.get("text", "").lower().strip()
            if word_text and (word_text in field_lower or field_lower in word_text):
                field_words.append(word_obj)

        if not field_words:
            # If no direct matches, try fuzzy matching based on character overlap
            for word_obj in ocr_words:
                word_text = word_obj.get("text", "").lower().strip()
                if word_text:
                    # Check if there's significant overlap between the field text and word text
                    # This handles cases where the LLM might have normalized the text
                    if self._has_significant_overlap(field_lower, word_text):
                        field_words.append(word_obj)

        if not field_words:
            # If still no matches, try matching individual words in the field text
            field_parts = field_lower.split()
            matched_words = set()
            for part in field_parts:
                if len(part) < 2:  # Skip very short parts
                    continue
                for word_obj in ocr_words:
                    word_text = word_obj.get("text", "").lower().strip()
                    if word_text and (part in word_text or word_text in part):
                        if word_obj not in matched_words:
                            matched_words.add(word_obj)

            field_words = list(matched_words)

        if not field_words:
            return 0.2  # Very low confidence if no matching words found

        # Calculate average confidence of matching words
        total_conf = sum(word_obj.get("confidence", 0.5) for word_obj in field_words)
        avg_conf = total_conf / len(field_words)

        # Boost confidence slightly if we have multiple matching words
        length_bonus = min(0.1, len(field_words) * 0.02)

        return min(1.0, avg_conf + length_bonus)

    def _has_significant_overlap(self, text1: str, text2: str) -> bool:
        """Check if two texts have significant character overlap"""
        if not text1 or not text2:
            return False

        # Remove common non-alphanumeric characters for comparison
        clean_text1 = re.sub(r'[^\w]', '', text1)
        clean_text2 = re.sub(r'[^\w]', '', text2)

        if not clean_text1 or not clean_text2:
            return False

        # Check if one text contains the other or if they share significant substrings
        if clean_text1 in clean_text2 or clean_text2 in clean_text1:
            return True

        # Check for common substrings of at least 3 characters
        min_substring_len = min(3, min(len(clean_text1), len(clean_text2)))
        for i in range(len(clean_text1) - min_substring_len + 1):
            for j in range(len(clean_text2) - min_substring_len + 1):
                if clean_text1[i:i+min_substring_len] == clean_text2[j:j+min_substring_len]:
                    return True

        return False

    def _calculate_completeness(self, field_text: str, expected_format: str = None) -> float:
        """Calculate confidence based on field completeness and format"""
        if not field_text or not field_text.strip():
            return 0.0

        # Basic completeness check
        text = field_text.strip()
        completeness = len(text) / max(len(text), 10)  # Normalize against expected min length
        completeness = min(1.0, completeness * 2)  # Boost for longer texts

        # Format-specific validation if expected format is provided
        if expected_format:
            if expected_format == "date":
                import re
                date_pattern = r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}-\d{2}-\d{2}"
                if re.search(date_pattern, text):
                    return min(1.0, completeness * 1.2)
                else:
                    return completeness * 0.5  # Penalty for wrong format
            elif expected_format == "currency":
                import re
                currency_pattern = r"\$?\d{1,3}(,\d{3})*\.?\d{2}|\d{1,3}(,\d{3})*\.?\d{2}\$?"
                if re.search(currency_pattern, text):
                    return min(1.0, completeness * 1.2)
                else:
                    return completeness * 0.5  # Penalty for wrong format
            elif expected_format == "alphanumeric_id":
                import re
                id_pattern = r"[A-Z0-9\-]{6,}"  # At least 6 alphanumeric chars/dashes
                if re.search(id_pattern, text.upper()):
                    return min(1.0, completeness * 1.2)
                else:
                    return completeness * 0.7  # Smaller penalty for IDs

        return completeness

    def is_confident_enough(self, overall_confidence: float, threshold: float = 0.75) -> bool:
        """Check if overall confidence meets the threshold for auto-approval"""
        return overall_confidence >= threshold