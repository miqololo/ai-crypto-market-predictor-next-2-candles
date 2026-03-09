"""Local vision service combining OCR with local LLM for prompt-based extraction with Armenian support."""
import io
import logging
from typing import Optional, Dict, Any
import httpx
from PIL import Image
import numpy as np

from app.config import get_settings
from app.services.ocr_service import get_ocr_service

logger = logging.getLogger(__name__)


class LocalVisionService:
    """
    Local vision service that combines OCR (with Armenian support) 
    with local LLM for prompt-based extraction.
    
    Strategy:
    1. Extract text using Tesseract OCR (supports Armenian)
    2. Send extracted text + prompt to local LLM
    3. LLM processes text according to prompt instructions
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.ocr_service = get_ocr_service()
    
    async def extract_with_prompt(
        self,
        image_data: bytes,
        prompt: str,
        languages: Optional[list] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract information from image using local OCR + LLM with custom prompt.
        
        Args:
            image_data: Image bytes
            prompt: Custom prompt describing what to extract
            languages: Language codes for OCR (default: ['en', 'es', 'hy'] for Armenian support)
            model: LLM model name (optional, uses default from config)
        
        Returns:
            Dictionary with extracted information and metadata
        """
        # Step 1: Extract text using OCR (supports Armenian via Tesseract)
        if languages is None:
            languages = ['en', 'es', 'hy']  # Default: English, Spanish, Armenian
        
        logger.info(f"Step 1: Extracting text with OCR (languages: {languages})")
        
        try:
            ocr_result = self.ocr_service.extract_text(
                image_data=image_data,
                languages=languages,
                engine='tesseract',  # Use Tesseract for Armenian support
                detail=0  # Simple text extraction
            )
            
            extracted_text = ocr_result.get('text', '')
            detected_languages = ocr_result.get('detected_languages', languages)
            
            if not extracted_text or len(extracted_text.strip()) < 5:
                logger.warning("OCR extracted very little text")
                return {
                    "text": "No text could be extracted from the image.",
                    "model": model or self.settings.llm_model,
                    "provider": "local",
                    "prompt": prompt,
                    "ocr_text": extracted_text,
                    "detected_languages": detected_languages
                }
            
            logger.info(f"OCR extracted {len(extracted_text)} characters")
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}", exc_info=True)
            raise RuntimeError(f"OCR extraction failed: {e}")
        
        # Step 2: Process extracted text with local LLM using prompt
        logger.info("Step 2: Processing text with local LLM")
        
        try:
            llm_result = await self._process_with_llm(
                extracted_text=extracted_text,
                prompt=prompt,
                model=model
            )
            
            return {
                "text": llm_result,
                "model": model or self.settings.llm_model,
                "provider": "local",
                "prompt": prompt,
                "ocr_text": extracted_text,  # Include raw OCR text
                "detected_languages": detected_languages,
                "method": "ocr+llm"
            }
        
        except Exception as e:
            logger.error(f"LLM processing failed: {e}", exc_info=True)
            # Fallback: return OCR text if LLM fails
            return {
                "text": extracted_text,
                "model": "ocr-only",
                "provider": "local",
                "prompt": prompt,
                "ocr_text": extracted_text,
                "detected_languages": detected_languages,
                "method": "ocr-only",
                "warning": f"LLM processing failed: {e}. Returning raw OCR text."
            }
    
    async def _process_with_llm(
        self,
        extracted_text: str,
        prompt: str,
        model: Optional[str]
    ) -> str:
        """Process extracted text with local LLM."""
        api_url = self.settings.llm_api_url or "http://localhost:11434"
        model_name = model or self.settings.llm_model
        
        if not api_url:
            raise ValueError("LLM_API_URL not configured. Set it in .env file")
        
        # Build system prompt for document processing
        system_prompt = """You are a document analysis assistant. Your task is to extract and format information from OCR-extracted text according to user instructions.

Guidelines:
- Extract only the requested information
- Format output clearly and structured
- Preserve important details like dates, amounts, names
- If information is not found, say so clearly
- For Armenian text, preserve the original Armenian characters
- Respond in the same language(s) as the document unless asked otherwise"""

        # Combine system prompt, user prompt, and OCR text
        user_message = f"""User Request: {prompt}

OCR Extracted Text:
{extracted_text}

Please extract and format the requested information from the OCR text above."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {
                "Content-Type": "application/json"
            }
            if self.settings.llm_api_key:
                headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
            
            # Determine endpoint based on API URL
            is_ollama = "ollama" in api_url.lower() or "localhost" in api_url or "127.0.0.1" in api_url
            
            if is_ollama:
                # Ollama uses /api/chat directly (not /v1/api/chat)
                # Strip /v1 if present in the URL
                base_url = api_url.rstrip("/v1").rstrip("/")
                endpoint = "/api/chat"
                request_data = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.1  # Lower temperature for more accurate extraction
                    }
                }
            else:
                # OpenAI-compatible API (may include /v1)
                base_url = api_url.rstrip("/")
                endpoint = "/chat/completions"
                request_data = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 4000
                }
            
            response = await client.post(
                f"{base_url}{endpoint}",
                json=request_data,
                headers=headers
            )
            
            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(f"LLM API error ({response.status_code}): {error_text}")
            
            result = response.json()
            
            # Extract content based on API format
            if "message" in result:
                content = result["message"]["content"]
            elif "choices" in result:
                content = result["choices"][0]["message"]["content"]
            else:
                raise RuntimeError(f"Unexpected LLM response format: {result}")
            
            return content


# Global instance
_local_vision_service: Optional[LocalVisionService] = None


def get_local_vision_service() -> LocalVisionService:
    """Get or create the global local vision service instance."""
    global _local_vision_service
    if _local_vision_service is None:
        _local_vision_service = LocalVisionService()
    return _local_vision_service
