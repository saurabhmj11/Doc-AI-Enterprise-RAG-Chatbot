"""
Structured Extractor Module

Extract structured shipment data from logistics documents.
"""

import json
import logging
from typing import Optional
try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None

from config import get_settings
from api.schemas import ShipmentData
from core.error_handling import (
    get_gemini_circuit_breaker,
    get_ollama_circuit_breaker,
    api_retry,
    APIError
)

settings = get_settings()
logger = logging.getLogger(__name__)


class StructuredExtractor:
    """
    Extract structured shipment data from logistics documents.
    
    Uses LLM with structured output prompt to extract:
    - shipment_id, shipper, consignee
    - pickup_datetime, delivery_datetime
    - equipment_type, mode, rate, currency
    - weight, carrier_name
    """
    
    EXTRACTION_PROMPT = """You are a logistics document data extractor. Extract the following fields from the document.

FIELDS TO EXTRACT:
1. shipment_id - Any shipment/load/order ID or reference number
2. shipper - The shipping company/sender name and address
3. consignee - The receiving company/destination name and address  
4. pickup_datetime - Pickup date and time (use ISO 8601 format if possible: YYYY-MM-DDTHH:MM:SS)
5. delivery_datetime - Delivery date and time (use ISO 8601 format if possible)
6. equipment_type - Type of equipment/trailer (e.g., "53' Dry Van", "Reefer", "Flatbed")
7. mode - Transportation mode (e.g., "TL", "LTL", "Intermodal", "Air")
8. rate - The total rate/cost as a number only (no currency symbol)
9. currency - Currency of the rate (e.g., "USD", "CAD", "EUR")
10. weight - Total weight with unit (e.g., "42,000 lbs", "19,000 kg")
11. carrier_name - Name of the carrier/trucking company

RULES:
- Return ONLY a valid JSON object
- Use null for any field not found in the document
- Do not guess or invent values - only extract what's explicitly stated
- For dates, try to use ISO 8601 format when possible
- For rate, extract only the numeric value (e.g., 2500.00 not "$2,500.00")

DOCUMENT:
{document_text}

Return ONLY valid JSON (no markdown, no explanation):"""

    def __init__(self):
        self.llm_mode = settings.llm_mode
        
        if self.llm_mode == "offline":
            import ollama
            self.ollama_client = ollama.Client(host=settings.ollama_base_url)
            self.ollama_model = settings.ollama_model
            self.llm = True # marker
            self.circuit_breaker = get_ollama_circuit_breaker()
            logger.info(f"Structured Extractor initialized in OFFLINE mode ({self.ollama_model})")
        else:
            # Configure Gemini
            if settings.gemini_api_key:
                genai.configure(api_key=settings.gemini_api_key)
                self.llm = genai.GenerativeModel('gemini-1.5-flash')
                self.circuit_breaker = get_gemini_circuit_breaker()
                logger.info("Structured Extractor initialized in ONLINE mode (Gemini)")
            else:
                self.llm = None
                self.circuit_breaker = None
                logger.warning("Structured Extractor initialized WITHOUT LLM (no API key)")
    
    def extract(self, document_text: str) -> tuple[ShipmentData, float]:
        """
        Extract structured data from document.
        
        Args:
            document_text: Full document text
            
        Returns:
            Tuple of (ShipmentData, confidence_score)
        """

        if not self.llm:
            logger.warning("LLM not available, using regex fallback extraction")
            return self._fallback_extraction(document_text)
        
        prompt = self.EXTRACTION_PROMPT.format(document_text=document_text[:8000])  # Limit context
        
        return self._extract_with_llm(prompt, document_text)

    def extract_offline(self, document_text: str) -> tuple[ShipmentData, float]:
        """
        Extract data using ONLY regex patterns (no LLM).
        Useful for fallback scenarios or when offline.
        """
        return self._fallback_extraction(document_text)

    def _extract_with_llm(self, prompt: str, document_text: str) -> tuple[ShipmentData, float]:
        """Internal method to handle LLM extraction logic."""
        try:
            # Wrapper for circuit breaker and retry
            @api_retry(api_name=f"{self.llm_mode.upper()}_Extraction", max_attempts=settings.retry_attempts)
            def _call_llm():
                if self.llm_mode == "offline":
                    response = self.ollama_client.chat(model=self.ollama_model, messages=[
                        {'role': 'user', 'content': prompt},
                    ])
                    return response['message']['content']
                else:
                    # Gemini
                    response = self.llm.generate_content(prompt)
                    if not response.text:
                        raise APIError("Empty response from Gemini")
                    return response.text.strip()
            
            # Call through circuit breaker
            json_text = self.circuit_breaker.call(_call_llm)
            logger.debug("LLM extraction completed successfully")
            
            # Clean up response (remove markdown code blocks if present)
            if json_text.startswith("```"):
                # Handle cases where the language name is included e.g. ```json
                lines = json_text.split('\n')
                if lines[0].startswith("```"):
                     json_text = "\n".join(lines[1:-1])
                else:
                     json_text = json_text.split("```")[1]
            
            json_text = json_text.strip()
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Create ShipmentData with validation
            shipment = ShipmentData(
                shipment_id=data.get("shipment_id"),
                shipper=data.get("shipper"),
                consignee=data.get("consignee"),
                pickup_datetime=data.get("pickup_datetime"),
                delivery_datetime=data.get("delivery_datetime"),
                equipment_type=data.get("equipment_type"),
                mode=data.get("mode"),
                rate=self._parse_rate(data.get("rate")),
                currency=data.get("currency"),
                weight=data.get("weight"),
                carrier_name=data.get("carrier_name")
            )
            
            # Calculate confidence based on fields found
            confidence = self._calculate_confidence(shipment)
            logger.info(f"LLM extraction completed with {confidence:.2f} confidence")
            
            return shipment, confidence
            
        except APIError as e:
            # Circuit breaker is open
            logger.warning(f"Circuit breaker open for extraction: {str(e)}")
            return self._fallback_extraction(document_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error in extractor: {e}\nOutput: {json_text[:200]}...")
            return self._fallback_extraction(document_text)
            
        except Exception as e:
            logger.error(f"Error in extractor: {type(e).__name__}: {str(e)}", exc_info=True)
            return self._fallback_extraction(document_text)
    
    def _parse_rate(self, rate_value) -> Optional[float]:
        """Parse rate value to float."""
        if rate_value is None:
            return None
        
        if isinstance(rate_value, (int, float)):
            return float(rate_value)
        
        if isinstance(rate_value, str):
            # Remove currency symbols and commas
            cleaned = rate_value.replace("$", "").replace(",", "").replace(" ", "")
            try:
                return float(cleaned)
            except ValueError:
                return None
        
        return None
    
    def _calculate_confidence(self, shipment: ShipmentData) -> float:
        """Calculate extraction confidence based on fields found."""
        fields = [
            shipment.shipment_id,
            shipment.shipper,
            shipment.consignee,
            shipment.pickup_datetime,
            shipment.delivery_datetime,
            shipment.equipment_type,
            shipment.mode,
            shipment.rate,
            shipment.currency,
            shipment.weight,
            shipment.carrier_name
        ]
        
        found_count = sum(1 for f in fields if f is not None)
        total_count = len(fields)
        
        # Base confidence on percentage of fields found
        # Weight important fields more heavily
        important_fields = [
            shipment.shipper,
            shipment.consignee,
            shipment.pickup_datetime,
            shipment.rate
        ]
        important_found = sum(1 for f in important_fields if f is not None)
        
        base_confidence = found_count / total_count
        important_bonus = (important_found / len(important_fields)) * 0.2
        
        return min(1.0, base_confidence + important_bonus)
    
    def _fallback_extraction(self, document_text: str) -> tuple[ShipmentData, float]:
        """Enhanced fallback extraction using comprehensive regex patterns."""
        import re
        
        logger.info("Using regex-based fallback extraction")
        shipment = ShipmentData()
        extracted_fields = []
        
        # Shipment ID patterns - more comprehensive
        shipment_id_patterns = [
            r'(?:Reference|Ref|Order|Load)(?:\s+ID|#)?[:\s\n]+([A-Z0-9-]{4,20})\b',
            r'\b(LD\d{4,10})\b', # Specific format if known
            r'(?:BOL|Bill of Lading|B/L)[#:\s]+([A-Z0-9-]+)',
        ]
        for pattern in shipment_id_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE | re.MULTILINE)
            if match:
                shipment.shipment_id = match.group(1).strip()
                extracted_fields.append('shipment_id')
                logger.debug(f"Extracted shipment_id: {shipment.shipment_id}")
                break
        
        # Shipper/Consignee patterns
        # Handle merged headers like "Shipper Consignee" common in bad PDF extractions
        # Shipper patterns
        shipper_patterns = [
            # Strict Pickup block looking for name after potential table headers
            r'Pickup\s*\n(?:^\s*\d+\s*$\n)?(?:^S\.No.*\n)?(?:^Commodity.*\n)?(?:^Weight.*\n)?(?:^Quantity.*\n)?(?:^Shipping Date.*\n)?\s*(?!(?:S\.No|Commodity|Weight|Quantity|Shipping Date|Date/Time))([A-Za-z][^\n]{3,50})\n',
            r'Shipper\s*:\s*([^\n]{5,100})', # Strict colon
            r'Ack\.\s+Shipper[:\s]+([^\n]{10,100})',
            r'Origin[:\s]+([^\n]{10,100})'
        ]
        for pattern in shipper_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE | re.MULTILINE)
            if match:
                val = match.group(1).strip()
                # Cleanup common artifacts
                if "Consignee" in val[:15] or "location" in val.lower(): 
                    continue
                val = re.sub(r'^\d+\.\s*', '', val) # Remove leading "1. "
                shipment.shipper = val
                extracted_fields.append('shipper')
                logger.debug(f"Extracted shipper: {shipment.shipper[:50]}...")
                break
        
        # Consignee patterns
        consignee_patterns = [
            # Strict Drop/Delivery block
            r'(?:Drop|Delivery)\s*\n(?:^\s*\d+\s*$\n)?(?:^S\.No.*\n)?(?:^Commodity.*\n)?(?:^Weight.*\n)?(?:^Quantity.*\n)?(?:^Delivery Date.*\n)?\s*(?!(?:S\.No|Commodity|Weight|Quantity|Delivery Date|Date/Time|Description))([A-Za-z][^\n]{3,50})\n',
            r'Consignee\s*:\s*([^\n]{5,100})', # Strict colon
            r'To\s*:\s*([^\n]{10,100})',
            r'Destination\s*:\s*([^\n]{10,100})'
        ]
        for pattern in consignee_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE | re.MULTILINE)
            if match:
                val = match.group(1).strip()
                val = re.sub(r'^\d+\.\s*', '', val)
                if "location" in val.lower():
                    continue
                shipment.consignee = val
                extracted_fields.append('consignee')
                logger.debug(f"Extracted consignee: {shipment.consignee[:50]}...")
                break
        
        # Rate patterns - improved
        rate_patterns = [
            r'(?:Rate|Total|Amount|Freight Charges)[:\s]*\$?\s*([\d,]+\.?\d*)',
            r'\$\s*([\d,]+\.\d{2})',
            r'USD[:\s]*([\d,]+\.?\d*)'
        ]
        for pattern in rate_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                rate_str = re.sub(r'[^\d.]', '', match.group(1))
                try:
                    shipment.rate = float(rate_str)
                    shipment.currency = "USD"
                    extracted_fields.extend(['rate', 'currency'])
                    logger.debug(f"Extracted rate: ${shipment.rate}")
                    break
                except ValueError:
                    pass
        
        # Date patterns - improved
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?:\s+\d{1,2}:\d{2}(?:\s*[AP]M)?)?'
        dates = re.findall(date_pattern, document_text)
        if len(dates) >= 1:
            shipment.pickup_datetime = dates[0]
            extracted_fields.append('pickup_datetime')
            logger.debug(f"Extracted pickup_datetime: {shipment.pickup_datetime}")
        if len(dates) >= 2:
            shipment.delivery_datetime = dates[1]
            extracted_fields.append('delivery_datetime')
            logger.debug(f"Extracted delivery_datetime: {shipment.delivery_datetime}")
        
        # Weight patterns - improved
        weight_pattern = r'(?:Weight[:\s]*)?([\d,]+)\s*(lbs?|kg|pounds?|kilograms?)'
        weight_match = re.search(weight_pattern, document_text, re.IGNORECASE)
        if weight_match:
            shipment.weight = f"{weight_match.group(1)} {weight_match.group(2)}"
            extracted_fields.append('weight')
            logger.debug(f"Extracted weight: {shipment.weight}")
        
        # Equipment type patterns
        equipment_patterns = [
            r'(?:Equipment|Trailer)[:\s]*([^\n]{5,50})',
            r'(\d{2,3}[\'"]?\s*(?:Dry Van|Reefer|Flatbed|Container))'
        ]
        for pattern in equipment_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                shipment.equipment_type = match.group(1).strip()
                extracted_fields.append('equipment_type')
                logger.debug(f"Extracted equipment_type: {shipment.equipment_type}")
                break
        
        # Mode patterns
        mode_pattern = r'\b(TL|LTL|FTL|LCL|FCL|Air|Ocean|Intermodal)\b'
        mode_match = re.search(mode_pattern, document_text, re.IGNORECASE)
        if mode_match:
            shipment.mode = mode_match.group(1).upper()
            extracted_fields.append('mode')
            logger.debug(f"Extracted mode: {shipment.mode}")
        
        # Carrier name patterns
        carrier_patterns = [
            r'Carrier[:\s]+([^\n]{5,50})',
            r'Trucking Company[:\s]+([^\n]{5,50})'
        ]
        for pattern in carrier_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                shipment.carrier_name = match.group(1).strip()
                extracted_fields.append('carrier_name')
                logger.debug(f"Extracted carrier_name: {shipment.carrier_name}")
                break
        
        confidence = self._calculate_confidence(shipment)
        logger.info(f"Regex extraction completed: {len(extracted_fields)} fields extracted, confidence={confidence:.2f}")
        
        return shipment, confidence


# Singleton instance
_extractor: Optional[StructuredExtractor] = None


def get_structured_extractor() -> StructuredExtractor:
    """Get or create structured extractor singleton."""
    global _extractor
    if _extractor is None:
        _extractor = StructuredExtractor()
    return _extractor
