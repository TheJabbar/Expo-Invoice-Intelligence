import re
from datetime import datetime
from loguru import logger
from api.services.confidence import ConfidenceCalculator
from api.services.llm_ocr_postprocessor import LLMOCRPostProcessor

class FieldExtractor:
    PATTERNS = {
        "invoice_no": [
            r"(?:^|\s)invoice\s*#?\s*[:-]?\s*([A-Z0-9\-]{6,})",
            r"(?:^|\s)inv\s*#?\s*[:-]?\s*([A-Z0-9\-]{6,})",
            r"(?:^|\s)no\s*[:-]?\s*([A-Z0-9\-]{6,})"
        ],
        "invoice_date": [
            r"(?:^|\s)date\s*[:-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
            r"(?:^|\s)invoice\s*date\s*[:-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
            r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\s+invoice"
        ],
        "total": [
            r"(?:^|\s)total\s*[:-]?\s*\$?([\d,]+\.\d{2})",
            r"(?:^|\s)amount\s*due\s*[:-]?\s*\$?([\d,]+\.\d{2})",
            r"(?:^|\s)balance\s*due\s*[:-]?\s*\$?([\d,]+\.\d{2})"
        ],
        "tax": [
            r"(?:^|\s)tax\s*[:-]?\s*\$?([\d,]+\.\d{2})",
            r"(?:^|\s)vat\s*[:-]?\s*\$?([\d,]+\.\d{2})",
            r"(?:^|\s)gst\s*[:-]?\s*\$?([\d,]+\.\d{2})"
        ]
    }

    def __init__(self):
        self.conf_calc = ConfidenceCalculator()
        self.llm_processor = LLMOCRPostProcessor()

    def extract(self, ocr_result: dict) -> dict:
        text = ocr_result["raw_text"].lower()
        words = ocr_result["words"]
        fields = {}
        confidences = {}

        # First, try LLM-based extraction for more accurate results
        try:
            llm_extracted = self.llm_processor.process_raw_ocr(ocr_result)
            logger.debug(f"LLM extracted fields: {llm_extracted}")
            # Use LLM results as primary, but fall back to regex for missing fields
            fields["vendor"] = llm_extracted.get("vendor") or self._extract_vendor(ocr_result)[0]
            fields["invoice_no"] = llm_extracted.get("invoice_no") or self._extract_with_patterns(text, words, self.PATTERNS["invoice_no"])[0]
            fields["invoice_date"] = llm_extracted.get("invoice_date") or self._extract_with_patterns(text, words, self.PATTERNS["invoice_date"])[0]
            fields["total"] = llm_extracted.get("total") or self._extract_with_patterns(text, words, self.PATTERNS["total"])[0]
            fields["tax"] = llm_extracted.get("tax") or self._extract_with_patterns(text, words, self.PATTERNS["tax"])[0]
            fields["debit_account"] = llm_extracted.get("debit_account") or self._map_account(fields.get("vendor", ""))
            fields["credit_account"] = llm_extracted.get("credit_account")  # Don't provide default here

            # Calculate confidences based on source
            # For LLM-extracted fields, use the confidence calculated by the LLM processor
            for field_name in ["vendor", "invoice_no", "invoice_date", "total", "tax", "debit_account", "credit_account"]:
                if llm_extracted.get(field_name) is not None:
                    # LLM-extracted field - use field-specific LLM confidence
                    field_llm_confidences = llm_extracted.get("field_llm_confidences", {})
                    if field_name in field_llm_confidences:
                        confidences[field_name] = field_llm_confidences[field_name]
                    else:
                        # Fallback to OCR-based confidence calculation if LLM didn't provide confidence
                        llm_value = llm_extracted.get(field_name)
                        if field_name in ["invoice_no", "invoice_date", "total", "tax"]:
                            expected_format = self._get_expected_format(field_name)
                            confidences[field_name] = self.conf_calc.calculate_field_confidence(
                                ocr_words=words,
                                field_text=str(llm_value) if llm_value else "",
                                pattern_matched=True,  # Consider LLM extraction as pattern matched
                                expected_format=expected_format
                            )
                        elif field_name == "vendor":
                            # Calculate OCR confidence for the vendor text
                            confidences[field_name] = self.conf_calc.calculate_field_confidence(
                                ocr_words=words,
                                field_text=str(llm_value) if llm_value else "",
                                pattern_matched=True,
                                expected_format=None
                            )
                        elif field_name in ["debit_account", "credit_account"]:
                            confidences[field_name] = 0.8  # Account mapping is usually reliable
                else:
                    # Fallback to regex-based confidence
                    if field_name in ["invoice_no", "invoice_date", "total", "tax"]:
                        expected_format = self._get_expected_format(field_name)
                        confidences[field_name] = self.conf_calc.calculate_field_confidence(
                            ocr_words=words,
                            field_text=str(fields[field_name]) if fields[field_name] else "",
                            pattern_matched=True,
                            expected_format=expected_format
                        )
                    elif field_name == "vendor":
                        _, vendor_conf = self._extract_vendor(ocr_result)
                        confidences[field_name] = vendor_conf
                    elif field_name == "debit_account":
                        confidences[field_name] = 0.8  # Account mapping is usually reliable
                    elif field_name == "credit_account":
                        confidences[field_name] = 0.0  # Default to 0.0 if not provided by LLM
        except Exception as e:
            logger.error(f"Error in LLM post-processing, falling back to regex: {e}")
            # Fallback to original regex-based extraction
            self._extract_with_regex_fallback(text, words, fields, confidences)
            # For fallback case, use traditional confidence calculation
            llm_extracted = {"field_llm_confidences": {}}  # Empty dict so the LLM confidence won't be used
            # Set credit_account to None with 0.0 confidence for fallback
            fields["credit_account"] = None
            confidences["credit_account"] = 0.0

        # Ensure all required fields exist
        required_fields = ["vendor", "invoice_no", "invoice_date", "total", "tax", "debit_account", "credit_account"]
        for field in required_fields:
            if field not in fields:
                fields[field] = None
            if field not in confidences:
                confidences[field] = 0.0

        # Calculate overall confidence considering only fields that were actually extracted
        if 'field_llm_confidences' in llm_extracted and llm_extracted['field_llm_confidences']:
            # Use the field-specific LLM confidences for fields that were extracted by LLM
            llm_field_confs = llm_extracted['field_llm_confidences']
            if llm_field_confs:
                # Get the fields that were actually extracted by the LLM (not None)
                llm_extracted_fields = {}
                for field_name in ["vendor", "invoice_no", "invoice_date", "total", "tax", "debit_account"]:
                    if llm_extracted.get(field_name) is not None:
                        llm_extracted_fields[field_name] = llm_extracted[field_name]

                # Get confidences only for the fields that were actually extracted by LLM
                active_confidences = {}
                for field in llm_extracted_fields.keys():
                    if field in llm_field_confs:
                        active_confidences[field] = llm_field_confs[field]
                    elif field in confidences:  # fallback to calculated confidences
                        active_confidences[field] = confidences[field]

                # Include credit_account in active confidences
                active_confidences["credit_account"] = confidences["credit_account"]

                if active_confidences:
                    # Use the minimum confidence among extracted fields as overall confidence
                    overall_conf = min(active_confidences.values())
                else:
                    # If no LLM confidences are available for extracted fields, use traditional calculation
                    overall_conf = self.conf_calc.calculate_overall_confidence(confidences)
            else:
                overall_conf = self.conf_calc.calculate_overall_confidence(confidences)
        else:
            # Include credit_account in the traditional calculation
            overall_conf = self.conf_calc.calculate_overall_confidence(confidences)

        return {
            "fields": fields,
            "confidence": overall_conf,
            "field_confidences": confidences
        }

    def _extract_with_regex_fallback(self, text, words, fields, confidences):
        """Fallback method using regex patterns when LLM processing fails"""
        # Extract vendor from top 20% of document (heuristic)
        vendor_text, vendor_conf = self._extract_vendor(text)
        fields["vendor"] = vendor_text
        confidences["vendor"] = vendor_conf

        # Extract other fields using patterns
        for field, patterns in self.PATTERNS.items():
            value, pattern_matched = self._extract_with_patterns(text, words, patterns)
            fields[field] = value

            # Calculate confidence for this field
            if value is not None:
                expected_format = self._get_expected_format(field)
                confidences[field] = self.conf_calc.calculate_field_confidence(
                    ocr_words=words,
                    field_text=str(value),
                    pattern_matched=pattern_matched,
                    expected_format=expected_format
                )
            else:
                confidences[field] = 0.0

        # Account mapping (simple rules)
        fields["debit_account"] = self._map_account(fields.get("vendor", ""))
        fields["credit_account"] = None  # Don't set a default value in fallback
        confidences["credit_account"] = 0.0  # No confidence in fallback for credit account

    def _get_expected_format(self, field_name: str) -> str:
        """Return expected format for confidence calculation"""
        format_map = {
            "invoice_date": "date",
            "total": "currency",
            "tax": "currency",
            "invoice_no": "alphanumeric_id"
        }
        return format_map.get(field_name)

    def _extract_vendor(self, ocr_result: dict) -> tuple:
        """Extract vendor from top portion of document"""
        h = ocr_result["image_shape"][0]
        top_words = [
            w for w in ocr_result["words"]
            if w["bbox"][0][1] < h * 0.2  # Top 20% of image
        ]
        if not top_words:
            return None, 0.3

        # Take first 3 words with highest confidence
        top_words.sort(key=lambda w: w["confidence"], reverse=True)
        vendor_text = " ".join([w["text"] for w in top_words[:3]])
        avg_conf = sum(w["confidence"] for w in top_words[:3]) / 3

        # Clean up vendor name
        vendor_text = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z\s\.\,\&\-]+$', '', vendor_text)
        return vendor_text.strip() or None, min(0.95, avg_conf)

    def _extract_with_patterns(self, text: str, words: list, patterns: list) -> tuple:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Get the matched text from the original OCR words to preserve casing
                matched_text = match.group(1).strip()

                # Try to find the original casing in the OCR words
                original_casing = self._find_original_casing(matched_text, words)
                if original_casing:
                    return self._normalize_value(original_casing), True
                else:
                    return self._normalize_value(matched_text), True
        return None, False

    def _find_original_casing(self, extracted_text: str, words: list) -> str:
        """
        Find the original casing of extracted text from OCR words
        """
        extracted_lower = extracted_text.lower()

        # Look for the text in OCR words
        for word_obj in words:
            word_text = word_obj.get("text", "")
            if extracted_lower in word_text.lower() or word_text.lower() in extracted_lower:
                return word_text

        # If not found directly, try to match parts
        extracted_parts = extracted_lower.split()
        matched_parts = []

        for word_obj in words:
            word_text = word_obj.get("text", "")
            for part in extracted_parts:
                if part in word_text.lower():
                    matched_parts.append(word_text)
                    break

        if matched_parts:
            return " ".join(matched_parts)

        return None


    def _normalize_value(self, value: str):
        # For invoice numbers, preserve the original format
        if re.match(r'[A-Z0-9\-]{6,}', value.upper()):
            return value.strip()

        # Try to parse as number first
        try:
            clean_val = re.sub(r"[^\d\.]", "", value)
            return float(clean_val)
        except:
            pass

        # Try to parse as date
        try:
            for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y"]:
                try:
                    return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
                except:
                    continue
        except:
            pass

        return value.strip()

    def _map_account(self, vendor: str) -> str:
        if not vendor:
            return "General Expense"

        vendor_lower = vendor.lower()
        # Check for IT services first to avoid false positives
        if any(k in vendor_lower for k in ["google", "aws", "microsoft", "cloud", "digitalocean", "web services", "software", "it services", "technology", "amazon web services"]):
            return "IT Services"
        elif any(k in vendor_lower for k in ["amazon", "staples", "office", "supply", "paper", "printer"]):
            return "Office Supplies"
        elif any(k in vendor_lower for k in ["rent", "lease", "property"]):
            return "Rent Expense"
        elif any(k in vendor_lower for k in ["electric", "water", "utility", "gas"]):
            return "Utilities"
        return "General Expense"