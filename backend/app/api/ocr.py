"""OCR API endpoints for document and image text extraction."""
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel

from app.services.ocr_service import get_ocr_service
from app.services.ocr_vision_service import get_vision_service
from app.services.ocr_local_vision_service import get_local_vision_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["ocr"])


class OCRResponse(BaseModel):
    """Response model for OCR extraction."""
    text: str
    lines: List[str]
    engine: str
    confidence: Optional[float] = None
    word_count: Optional[int] = None
    details: Optional[List[dict]] = None
    detected_languages: Optional[List[str]] = None
    auto_detected: Optional[bool] = None


class LanguagesResponse(BaseModel):
    """Response model for available languages."""
    available_languages: dict


class PromptExtractRequest(BaseModel):
    """Request model for prompt-based extraction."""
    prompt: str
    model: Optional[str] = None
    provider: str = "openai"  # openai, anthropic, google, ollama


class PromptExtractResponse(BaseModel):
    """Response model for prompt-based extraction."""
    text: str
    model: str
    provider: str
    prompt: str
    usage: Optional[dict] = None
    ocr_text: Optional[str] = None  # Raw OCR text (for local provider)
    detected_languages: Optional[List[str]] = None  # Detected languages
    method: Optional[str] = None  # Extraction method
    warning: Optional[str] = None  # Warnings


@router.post("/extract", response_model=OCRResponse)
async def extract_text(
    file: UploadFile = File(..., description="Image file to extract text from"),
    languages: Optional[str] = Query(None, description="Comma-separated language codes (e.g., 'en,es'). Leave empty for auto-detection if auto_detect=True"),
    engine: Optional[str] = Query(None, description="OCR engine: 'easyocr' or 'tesseract' (auto if not specified)"),
    detail: int = Query(0, ge=0, le=1, description="Detail level: 0=text only, 1=text with bounding boxes"),
    auto_detect: bool = Query(False, description="Automatically detect languages from the document. Ignores 'languages' parameter if True.")
):
    """
    Extract text from an uploaded image file.
    
    Supported formats: PNG, JPEG, JPG, GIF, BMP, TIFF, WebP
    
    **Languages**: 
    - EasyOCR: en, es, fr, de, it, pt, ru, zh, ja, ko, and many more
    - Tesseract: eng, spa, fra, deu, ita, por, rus, chi_sim, jpn, kor, **hye (Armenian)**, etc.
    
    **Armenian Support**: 
    - Use `engine=tesseract` with language code `hye` or `hy` (auto-converted)
    - Example: `languages=hye` or `languages=hy` (will be converted to `hye`)
    
    **Engines**:
    - easyocr: Modern, easy to use, supports many languages (default)
    - tesseract: Traditional OCR engine, requires system installation
      **Recommended on macOS ARM** to avoid OpenMP threading issues
    
    **macOS ARM Note**: If you experience crashes (segmentation faults), 
    use `engine=tesseract` parameter. EasyOCR may have threading issues on macOS ARM.
    
    **Auto-Detection**:
    - Set `auto_detect=true` to automatically detect languages from the document
    - Only detects between 3 supported languages: English (en), Spanish (es), and Armenian (hy)
    - First extracts with all 3 supported languages, detects languages from text, then re-extracts with detected languages
    - Returns `detected_languages` array in response
    
    **Multi-Language Documents**:
    - Specify multiple languages: `languages=en,es,fr` (comma-separated)
    - Or use `auto_detect=true` to automatically detect and extract all languages
    
    **Detail levels**:
    - 0: Returns only extracted text
    - 1: Returns text with bounding boxes, confidence scores, and word positions
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected an image file."
            )
        
        # Read file content
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Parse languages
        lang_list = None
        if languages:
            lang_list = [lang.strip() for lang in languages.split(',') if lang.strip()]
        
        # Get OCR service
        ocr_service = get_ocr_service()
        
        # Extract text
        result = ocr_service.extract_text(
            image_data=image_data,
            languages=lang_list if not auto_detect else None,
            engine=engine,
            detail=detail,
            auto_detect=auto_detect
        )
        
        return OCRResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        error_msg = str(e)
        # Provide helpful suggestion for macOS ARM users
        import platform
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            if 'segmentation' in error_msg.lower() or 'crash' in error_msg.lower():
                error_msg += " Try using engine=tesseract parameter to avoid threading issues."
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        logger.error(f"OCR extraction error: {e}", exc_info=True)
        error_msg = f"OCR extraction failed: {str(e)}"
        # Suggest Tesseract on macOS ARM if EasyOCR fails
        import platform
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            if engine != 'tesseract':
                error_msg += " Consider using engine=tesseract on macOS ARM to avoid threading issues."
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/languages", response_model=LanguagesResponse)
def get_available_languages():
    """
    Get list of available languages for each OCR engine.
    
    Returns available language codes for EasyOCR and Tesseract engines.
    """
    try:
        ocr_service = get_ocr_service()
        languages = ocr_service.get_available_languages()
        return LanguagesResponse(available_languages=languages)
    except Exception as e:
        logger.error(f"Error getting languages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health_check():
    """
    Check OCR service health and availability.
    
    Returns status of OCR engines and their availability.
    """
    try:
        ocr_service = get_ocr_service()
        status = {
            "status": "ok",
            "easyocr_available": ocr_service.easyocr_reader is not None,
            "tesseract_available": False,
            "preferred_engine": "easyocr" if ocr_service.prefer_easyocr else "tesseract"
        }
        
        # Check Tesseract availability
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            status["tesseract_available"] = True
        except:
            pass
        
        return status
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/extract-with-prompt", response_model=PromptExtractResponse)
async def extract_with_prompt(
    file: UploadFile = File(..., description="Image file to analyze"),
    prompt: str = Query(..., description="Custom prompt describing what to extract from the document"),
    model: Optional[str] = Query(None, description="Model name (optional, uses provider default)"),
    provider: str = Query("local", description="Vision model provider: local, openai, anthropic, google, or ollama"),
    languages: Optional[str] = Query(None, description="Comma-separated language codes for OCR (default: en,es,hy for Armenian support)")
):
    """
    Extract specific information from documents using advanced vision-language models with custom prompts.
    
    This endpoint uses AI vision models (GPT-4 Vision, Claude, Gemini, etc.) that can understand
    both images and natural language prompts to extract exactly what you need.
    
    **Supported Providers**:
    - **local** (RECOMMENDED for Armenian): OCR (Tesseract) + Local LLM
      - Uses Tesseract OCR with Armenian support, then processes with your local LLM
      - Supports Armenian, English, Spanish, and many other languages
      - Works completely offline
      - Requires: LLM_API_URL configured (e.g., Ollama at http://localhost:11434)
    - **openai**: GPT-4 Vision, GPT-4o (requires OPENAI_API_KEY or LLM_API_KEY)
    - **anthropic**: Claude 3.5 Sonnet, Claude 3 Opus (requires ANTHROPIC_API_KEY)
    - **google**: Gemini 1.5 Pro (requires GOOGLE_API_KEY)
    - **ollama**: LLaVA, BakLLaVA (requires local Ollama installation, limited Armenian support)
    
    **Example Prompts**:
    - "Extract all dates and amounts from this invoice"
    - "Find the total price and item names from this receipt"
    - "Extract the sender, recipient, and subject from this letter"
    - "List all names and email addresses from this document"
    - "Extract the table data and format it as JSON"
    - "What is the main topic and key points of this document?"
    
    **Advantages over traditional OCR**:
    - Understands document structure and context
    - Can extract structured data (tables, forms, etc.)
    - Follows natural language instructions
    - Better at handling complex layouts
    - Can answer questions about the document
    
    **Configuration**:
    Set API keys in `.env`:
    - `OPENAI_API_KEY=your_key` (for OpenAI)
    - `ANTHROPIC_API_KEY=your_key` (for Anthropic)
    - `GOOGLE_API_KEY=your_key` (for Google)
    - `LLM_API_URL=http://localhost:11434` (for Ollama)
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected an image file."
            )
        
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Read file content
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Validate provider
        valid_providers = ["local", "openai", "anthropic", "google", "ollama"]
        if provider not in valid_providers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {provider}. Must be one of: {', '.join(valid_providers)}"
            )
        
        # Parse languages if provided
        lang_list = None
        if languages:
            lang_list = [lang.strip() for lang in languages.split(',') if lang.strip()]
        
        # Use local service for Armenian support
        if provider == "local":
            local_service = get_local_vision_service()
            result = await local_service.extract_with_prompt(
                image_data=image_data,
                prompt=prompt,
                languages=lang_list,
                model=model
            )
        else:
            # Use cloud vision services
            vision_service = get_vision_service()
            result = await vision_service.extract_with_prompt(
                image_data=image_data,
                prompt=prompt,
                model=model,
                provider=provider
            )
        
        return PromptExtractResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Prompt-based extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
