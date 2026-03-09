"""OCR service for extracting text from images and documents."""
import io
import logging
import os
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

# Fix OpenMP threading issues on macOS ARM
# Set environment variables before importing libraries that use OpenMP
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    LangDetectException = Exception

logger = logging.getLogger(__name__)


class OCRService:
    """Service for OCR operations using EasyOCR and/or Tesseract."""
    
    def __init__(self, prefer_easyocr: bool = True):
        """
        Initialize OCR service.
        
        Args:
            prefer_easyocr: If True, use EasyOCR first (default). If False, prefer Tesseract.
                            On macOS ARM, Tesseract is recommended due to OpenMP issues.
        """
        self.prefer_easyocr = prefer_easyocr
        self.easyocr_reader = None
        self.easyocr_current_languages = None  # Track current languages for EasyOCR reader
        
        # On macOS ARM, prefer Tesseract by default to avoid OpenMP crashes
        import platform
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            if prefer_easyocr:
                logger.info("macOS ARM detected. Consider using Tesseract to avoid OpenMP threading issues.")
        
        self._initialize_readers()
    
    def _initialize_readers(self):
        """Initialize OCR readers."""
        if EASYOCR_AVAILABLE and self.prefer_easyocr:
            try:
                # Initialize EasyOCR reader (supports multiple languages)
                # First run will download models, so it might take a moment
                # Use single-threaded mode to avoid OpenMP crashes on macOS ARM
                logger.info("Initializing EasyOCR reader...")
                # Set quantize=False and set workers=1 to avoid threading issues
                self.easyocr_reader = easyocr.Reader(
                    ['en'], 
                    gpu=False,
                    quantize=False,
                    verbose=False
                )
                self.easyocr_current_languages = ['en']
                logger.info("EasyOCR reader initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
                # If EasyOCR fails, prefer Tesseract
                if TESSERACT_AVAILABLE:
                    logger.info("Falling back to Tesseract OCR")
                    self.prefer_easyocr = False
        
        if TESSERACT_AVAILABLE and not self.prefer_easyocr:
            try:
                # Test Tesseract availability
                pytesseract.get_tesseract_version()
                logger.info("Tesseract OCR is available")
            except Exception as e:
                logger.warning(f"Tesseract OCR not available: {e}")
    
    def extract_text(
        self,
        image_data: bytes,
        languages: Optional[List[str]] = None,
        engine: Optional[str] = None,
        detail: int = 0,
        auto_detect: bool = False
    ) -> Dict[str, any]:
        """
        Extract text from image.
        
        Args:
            image_data: Image bytes (PNG, JPEG, etc.)
            languages: List of language codes (e.g., ['en', 'es']). 
                       If None and auto_detect=False, defaults to ['en'].
                       If auto_detect=True, languages will be detected automatically.
            engine: OCR engine to use ('easyocr', 'tesseract', or None for auto)
            detail: Detail level (0=text only, 1=text with bounding boxes)
            auto_detect: If True, automatically detect languages from the image.
                        First extracts with common languages, then detects and re-extracts.
        
        Returns:
            Dictionary with extracted text and metadata, including detected_languages if auto_detect=True
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            # Determine which engine to use
            if engine is None:
                engine = self._select_engine()
            
            # Auto-detect languages if requested
            if auto_detect:
                return self._extract_with_auto_detect(image_array, engine, detail)
            
            # Use provided languages or default
            if languages is None:
                languages = ['en']
            
            if engine == 'easyocr' and self.easyocr_reader:
                return self._extract_with_easyocr(image_array, languages, detail)
            elif engine == 'tesseract' and TESSERACT_AVAILABLE:
                return self._extract_with_tesseract(image_array, languages, detail)
            else:
                raise ValueError(f"OCR engine '{engine}' is not available")
        
        except Exception as e:
            logger.error(f"Error extracting text: {e}", exc_info=True)
            raise
    
    def _select_engine(self) -> str:
        """Select the best available OCR engine."""
        if self.prefer_easyocr and self.easyocr_reader:
            return 'easyocr'
        elif TESSERACT_AVAILABLE:
            return 'tesseract'
        else:
            raise RuntimeError("No OCR engine available. Install easyocr or pytesseract.")
    
    def _detect_languages_from_text(self, text: str) -> List[str]:
        """
        Detect languages from extracted text.
        Only detects between supported languages: en, es, hy
        """
        if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 10:
            return ['en']
        
        # Supported languages for auto-detection
        supported_languages = ['en', 'ru', 'hy']
        
        try:
            # Get all detected languages with probabilities
            detected_langs = detect_langs(text)
            
            # Filter to only supported languages and get top 3 by probability
            supported_detected = [
                lang for lang in detected_langs 
                if lang.lang in supported_languages and lang.prob > 0.2
            ]
            
            if supported_detected:
                # Sort by probability and get top languages
                supported_detected.sort(key=lambda x: x.prob, reverse=True)
                result = [lang.lang for lang in supported_detected[:3]]
                logger.info(f"Detected supported languages: {result} with probabilities: {[f'{lang.prob:.2f}' for lang in supported_detected[:len(result)]]}")
                return result
            else:
                # Fallback: try primary detection
                try:
                    primary_lang = detect(text)
                    if primary_lang in supported_languages:
                        logger.info(f"Detected primary supported language: {primary_lang}")
                        return [primary_lang]
                    else:
                        logger.info(f"Detected language '{primary_lang}' not in supported set. Using all supported languages.")
                        return supported_languages
                except:
                    return supported_languages
        
        except (LangDetectException, Exception) as e:
            logger.warning(f"Language detection failed: {e}. Using all supported languages.")
            return supported_languages
    
    def _extract_with_auto_detect(self, image_array: np.ndarray, engine: str, detail: int) -> Dict[str, any]:
        """
        Extract text with automatic language detection.
        Only detects between 3 supported languages: English, Spanish, and Armenian.
        """
        # Only support 3 languages for auto-detection
        supported_languages = ['en', 'es', 'hy']  # English, Spanish, Armenian
        logger.info(f"Auto-detecting languages from supported set: {supported_languages}")
        
        try:
            # Step 1: Initial extraction with all 3 supported languages
            if engine == 'easyocr' and self.easyocr_reader:
                initial_result = self._extract_with_easyocr(image_array, supported_languages, detail)
            elif engine == 'tesseract' and TESSERACT_AVAILABLE:
                # Convert to Tesseract codes
                lang_map = {'en': 'eng', 'es': 'spa', 'hy': 'hye'}
                tesseract_langs = [lang_map.get(lang, lang) for lang in supported_languages]
                initial_result = self._extract_with_tesseract(image_array, tesseract_langs, detail)
            else:
                raise ValueError(f"OCR engine '{engine}' is not available")
            
            initial_text = initial_result.get('text', '')
            if not initial_text or len(initial_text.strip()) < 10:
                initial_result['detected_languages'] = ['en']
                initial_result['auto_detected'] = False
                return initial_result
            
            # Step 2: Detect languages from extracted text, but filter to only supported languages
            detected_langs_raw = self._detect_languages_from_text(initial_text)
            
            # Filter detected languages to only include our 3 supported ones
            detected_langs = [lang for lang in detected_langs_raw if lang in supported_languages]
            
            # If no supported languages detected, use all 3 as fallback
            if not detected_langs:
                logger.info("No supported languages detected, using all 3 supported languages")
                detected_langs = supported_languages
            
            # Step 3: Re-extract with detected languages (if different from initial)
            if set(detected_langs) != set(supported_languages):
                logger.info(f"Re-extracting with detected languages: {detected_langs}")
                try:
                    if engine == 'easyocr' and self.easyocr_reader:
                        final_result = self._extract_with_easyocr(image_array, detected_langs, detail)
                    elif engine == 'tesseract' and TESSERACT_AVAILABLE:
                        lang_map = {'en': 'eng', 'hy': 'hye', 'es': 'spa'}
                        tesseract_langs = [lang_map.get(lang, lang) for lang in detected_langs]
                        final_result = self._extract_with_tesseract(image_array, tesseract_langs, detail)
                    else:
                        final_result = initial_result
                    final_result['detected_languages'] = detected_langs
                    final_result['auto_detected'] = True
                    logger.info(f"Auto-detection complete. Detected languages: {detected_langs}")
                    return final_result
                except Exception as e:
                    logger.warning(f"Re-extraction with detected languages failed: {e}. Using initial result.")
            
            # If detected languages match all supported, return initial result
            initial_result['detected_languages'] = detected_langs
            initial_result['auto_detected'] = True
            logger.info(f"Auto-detection complete. Detected languages: {detected_langs}")
            return initial_result
        except Exception as e:
            logger.error(f"Auto-detection failed: {e}", exc_info=True)
            if engine == 'easyocr' and self.easyocr_reader:
                result = self._extract_with_easyocr(image_array, ['en'], detail)
            elif engine == 'tesseract' and TESSERACT_AVAILABLE:
                result = self._extract_with_tesseract(image_array, ['eng'], detail)
            else:
                raise
            result['detected_languages'] = ['en']
            result['auto_detected'] = False
            return result
    
    def _extract_with_easyocr(
        self,
        image_array: np.ndarray,
        languages: Optional[List[str]],
        detail: int
    ) -> Dict[str, any]:
        """Extract text using EasyOCR."""
        if languages is None:
            languages = ['en']
        
        # Update reader languages if needed
        # Check if we need to recreate the reader with different languages
        if self.easyocr_current_languages is None or set(languages) != set(self.easyocr_current_languages):
            try:
                logger.info(f"Creating EasyOCR reader with languages: {languages}")
                self.easyocr_reader = easyocr.Reader(
                    languages, 
                    gpu=False,
                    quantize=False,
                    verbose=False
                )
                self.easyocr_current_languages = languages
            except Exception as e:
                logger.warning(f"Failed to update EasyOCR reader: {e}")
                # Fallback to Tesseract if EasyOCR fails
                if TESSERACT_AVAILABLE:
                    logger.info("Falling back to Tesseract due to EasyOCR error")
                    return self._extract_with_tesseract(image_array, languages, detail)
                raise
        
        # Perform OCR with error handling for segmentation faults
        try:
            # Use paragraph=False to avoid some threading issues
            results = self.easyocr_reader.readtext(
                image_array, 
                detail=detail,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7
            )
        except (SystemError, OSError, RuntimeError) as e:
            # Catch potential segmentation faults and threading errors
            logger.error(f"EasyOCR extraction failed (possibly segfault): {e}")
            # Fallback to Tesseract if available
            if TESSERACT_AVAILABLE:
                logger.info("Falling back to Tesseract due to EasyOCR crash")
                return self._extract_with_tesseract(image_array, languages, detail)
            raise RuntimeError(f"OCR extraction failed: {e}. Try using Tesseract engine instead.")
        
        # Format results
        if detail == 0:
            # Simple text extraction
            text_lines = [result[1] for result in results]
            full_text = '\n'.join(text_lines)
            return {
                'text': full_text,
                'lines': text_lines,
                'engine': 'easyocr',
                'confidence': None  # EasyOCR doesn't provide confidence in detail=0 mode
            }
        else:
            # Detailed extraction with bounding boxes
            text_data = []
            for (bbox, text, confidence) in results:
                text_data.append({
                    'text': text,
                    'confidence': float(confidence),
                    'bbox': bbox.tolist() if isinstance(bbox, np.ndarray) else bbox
                })
            
            full_text = '\n'.join([item['text'] for item in text_data])
            avg_confidence = np.mean([item['confidence'] for item in text_data]) if text_data else 0.0
            
            return {
                'text': full_text,
                'lines': [item['text'] for item in text_data],
                'details': text_data,
                'engine': 'easyocr',
                'confidence': float(avg_confidence),
                'word_count': len(full_text.split())
            }
    
    def _extract_with_tesseract(
        self,
        image_array: np.ndarray,
        languages: Optional[List[str]],
        detail: int
    ) -> Dict[str, any]:
        """Extract text using Tesseract OCR."""
        if languages is None:
            languages = ['eng']  # Tesseract uses 'eng' not 'en'
        else:
            # Convert common language codes to Tesseract codes
            lang_map = {
                'en': 'eng',
                'hy': 'hye',  # Armenian
                'es': 'spa',
                'fr': 'fra',
                'de': 'deu',
                'it': 'ita',
                'pt': 'por',
                'ru': 'rus',
                'zh': 'chi_sim',  # Simplified Chinese
                'ja': 'jpn',
                'ko': 'kor',
                'ar': 'ara',
                'hi': 'hin',
                'th': 'tha',
                'vi': 'vie',
            }
            languages = [lang_map.get(lang.lower(), lang) for lang in languages]
        
        lang_string = '+'.join(languages)
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array)
        else:
            image = image_array
        
        if detail == 0:
            # Simple text extraction
            text = pytesseract.image_to_string(image, lang=lang_string)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return {
                'text': text,
                'lines': lines,
                'engine': 'tesseract',
                'confidence': None
            }
        else:
            # Detailed extraction with bounding boxes
            data = pytesseract.image_to_data(image, lang=lang_string, output_type=Output.DICT)
            
            text_data = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text:  # Only include non-empty text
                    text_data.append({
                        'text': text,
                        'confidence': float(data['conf'][i]) if data['conf'][i] != -1 else None,
                        'bbox': [
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ],
                        'level': data['level'][i],
                        'page_num': data['page_num'][i],
                        'block_num': data['block_num'][i],
                        'par_num': data['par_num'][i],
                        'line_num': data['line_num'][i],
                        'word_num': data['word_num'][i]
                    })
            
            full_text = '\n'.join([item['text'] for item in text_data])
            confidences = [item['confidence'] for item in text_data if item['confidence'] is not None]
            avg_confidence = np.mean(confidences) if confidences else None
            
            return {
                'text': full_text,
                'lines': [item['text'] for item in text_data],
                'details': text_data,
                'engine': 'tesseract',
                'confidence': float(avg_confidence) if avg_confidence is not None else None,
                'word_count': len(full_text.split())
            }
    
    def extract_from_file(
        self,
        file_path: str,
        languages: Optional[List[str]] = None,
        engine: Optional[str] = None,
        detail: int = 0,
        auto_detect: bool = False
    ) -> Dict[str, any]:
        """
        Extract text from a file path.
        
        Args:
            file_path: Path to image file
            languages: List of language codes
            engine: OCR engine to use
            detail: Detail level
            auto_detect: If True, automatically detect languages
        
        Returns:
            Dictionary with extracted text and metadata
        """
        with open(file_path, 'rb') as f:
            image_data = f.read()
        return self.extract_text(image_data, languages, engine, detail, auto_detect)
    
    def get_available_languages(self) -> Dict[str, List[str]]:
        """Get list of available languages for each OCR engine."""
        result = {}
        
        if self.easyocr_reader:
            # EasyOCR supports many languages (note: Armenian 'hy' may not be fully supported)
            result['easyocr'] = [
                'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko',
                'ar', 'hi', 'th', 'vi', 'id', 'tr', 'pl', 'nl', 'cs', 'sv',
                'fi', 'no', 'da', 'el', 'he', 'uk', 'bg', 'hr', 'sk', 'sl',
                'ro', 'hu', 'et', 'lv', 'lt', 'mt', 'ga', 'cy', 'is', 'mk',
                'sq', 'sr', 'bs', 'me', 'be', 'ka', 'az', 'kk', 'ky',
                'uz', 'mn', 'ne', 'si', 'my', 'km', 'lo', 'am', 'ti',
                'sw', 'zu', 'af', 'eu', 'ca', 'gl', 'br', 'gd', 'ga', 'cy'
            ]
            result['easyocr_note'] = 'Armenian (hy) is not fully supported in EasyOCR. Use Tesseract with "hye" instead.'
        
        if TESSERACT_AVAILABLE:
            try:
                langs = pytesseract.get_languages()
                result['tesseract'] = langs
                # Add note about Armenian if available
                if 'hye' in langs:
                    result['tesseract_note'] = 'Armenian is supported with language code "hye"'
            except Exception as e:
                logger.warning(f"Could not get Tesseract languages: {e}")
                result['tesseract'] = []
        
        return result


# Global instance
_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Get or create the global OCR service instance."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
