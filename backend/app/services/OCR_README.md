# OCR Service Documentation

The OCR service provides text extraction capabilities from images and documents using EasyOCR and/or Tesseract OCR engines.

## Features

- **Multiple OCR Engines**: Supports EasyOCR (default) and Tesseract
- **Multi-language Support**: Extract text in many languages
- **Detail Levels**: Get simple text or detailed results with bounding boxes
- **File Upload API**: RESTful endpoints for document processing

## Installation

### Required Dependencies

The OCR dependencies are already in `requirements.txt`:
- `easyocr>=1.7.0` - Modern OCR library (recommended)
- `pytesseract>=0.3.10` - Python wrapper for Tesseract OCR
- `Pillow>=10.0.0` - Image processing

### System Requirements

**For EasyOCR** (recommended):
- No additional system dependencies needed
- First run will download models automatically (~500MB)
- Works on CPU, GPU support available if CUDA is installed

**For Tesseract** (optional):
- Requires Tesseract OCR to be installed on your system:
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

### API Endpoints

#### 1. Extract Text from Image

```bash
POST /api/ocr/extract
Content-Type: multipart/form-data

Parameters:
- file: Image file (PNG, JPEG, GIF, BMP, TIFF, WebP)
- languages: Optional comma-separated language codes (e.g., "en,es")
- engine: Optional OCR engine ("easyocr" or "tesseract")
- detail: 0 for text only, 1 for text with bounding boxes
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/ocr/extract?detail=1" \
  -F "file=@document.png" \
  -F "languages=en"
```

**Example using Python:**
```python
import requests

with open('document.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/ocr/extract',
        files={'file': f},
        params={'languages': 'en', 'detail': 1}
    )
    result = response.json()
    print(result['text'])
    print(f"Confidence: {result['confidence']}")
```

#### 2. Get Available Languages

```bash
GET /api/ocr/languages
```

Returns available languages for each OCR engine.

#### 3. Health Check

```bash
GET /api/ocr/health
```

Check OCR service status and engine availability.

### Python Service Usage

```python
from app.services.ocr_service import get_ocr_service

# Get OCR service instance
ocr_service = get_ocr_service()

# Extract text from file
result = ocr_service.extract_from_file(
    'document.png',
    languages=['en'],
    detail=1
)

print(result['text'])
print(f"Confidence: {result['confidence']}")

# Extract text from bytes
with open('image.jpg', 'rb') as f:
    image_data = f.read()

result = ocr_service.extract_text(
    image_data=image_data,
    languages=['en', 'es'],  # Multi-language
    engine='easyocr',
    detail=0  # Simple text extraction
)

print(result['text'])
```

## Supported Languages

### EasyOCR (Default)
Supports 80+ languages including:
- English (en), Spanish (es), French (fr), German (de)
- Chinese (zh), Japanese (ja), Korean (ko)
- Arabic (ar), Hindi (hi), Thai (th)
- And many more...

### Tesseract
Supports 100+ languages. Language codes differ from EasyOCR:
- English: `eng`
- Spanish: `spa`
- French: `fra`
- Chinese Simplified: `chi_sim`
- Japanese: `jpn`
- Korean: `kor`

## Response Format

### Simple Extraction (detail=0)
```json
{
  "text": "Extracted text here...",
  "lines": ["Line 1", "Line 2", "Line 3"],
  "engine": "easyocr",
  "confidence": null
}
```

### Detailed Extraction (detail=1)
```json
{
  "text": "Extracted text here...",
  "lines": ["Line 1", "Line 2"],
  "engine": "easyocr",
  "confidence": 0.95,
  "word_count": 10,
  "details": [
    {
      "text": "Word",
      "confidence": 0.98,
      "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ]
}
```

## Performance Tips

1. **First Run**: EasyOCR downloads models on first use (~500MB). Subsequent runs are faster.
2. **GPU Acceleration**: If you have CUDA installed, EasyOCR will automatically use GPU for faster processing.
3. **Image Preprocessing**: For better results, preprocess images:
   - Convert to grayscale
   - Increase contrast
   - Remove noise
   - Ensure good resolution (300 DPI recommended)
4. **Language Selection**: Specify languages to improve accuracy and speed.

## Examples

### Extract Text from Receipt
```python
result = ocr_service.extract_from_file('receipt.jpg', languages=['en'])
print(result['text'])
```

### Multi-language Document
```python
result = ocr_service.extract_text(
    image_data=image_bytes,
    languages=['en', 'es'],  # English and Spanish
    detail=1
)
```

### Get Word Positions
```python
result = ocr_service.extract_text(image_data, detail=1)
for word_detail in result['details']:
    print(f"{word_detail['text']}: {word_detail['bbox']}")
```

## Troubleshooting

### EasyOCR Not Working
- Check internet connection (first run needs to download models)
- Ensure sufficient disk space (~500MB for models)
- Check logs for specific error messages

### Tesseract Not Found
- Install Tesseract OCR on your system
- On macOS: `brew install tesseract`
- Verify installation: `tesseract --version`

### Low Accuracy
- Use higher resolution images (300+ DPI)
- Preprocess images (grayscale, contrast adjustment)
- Specify correct languages
- Try different OCR engine

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

The OCR endpoints will be available under the `/api/ocr` section.
