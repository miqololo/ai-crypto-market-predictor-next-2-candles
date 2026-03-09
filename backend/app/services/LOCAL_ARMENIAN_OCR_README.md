# Local OCR with Armenian Support + Prompt-Based Extraction

This service combines **Tesseract OCR** (with Armenian support) with your **local LLM** to provide prompt-based document extraction that works completely offline and supports Armenian.

## How It Works

1. **Step 1: OCR Extraction**
   - Uses Tesseract OCR with Armenian language support (`hye`)
   - Extracts text from image in multiple languages (default: English, Spanish, Armenian)
   - Supports: `en`, `es`, `hy` (and many more via Tesseract)

2. **Step 2: LLM Processing**
   - Sends extracted text + your custom prompt to your local LLM
   - LLM processes the text according to your instructions
   - Returns structured, formatted results

## Setup

### 1. Install Tesseract with Armenian Support

**macOS:**
```bash
brew install tesseract tesseract-lang
# This installs all language packs including Armenian (hye)
```

**Verify Armenian support:**
```bash
tesseract --list-langs | grep hye
# Should output: hye
```

### 2. Configure Local LLM

Set in `.env`:
```env
# Your local LLM endpoint (Ollama, vLLM, etc.)
LLM_API_URL=http://localhost:11434

# Optional: API key if needed
LLM_API_KEY=

# Model name (e.g., qwen2.5-7b-instruct, llama3, etc.)
LLM_MODEL=qwen2.5-7b-instruct
```

### 3. Start Your Local LLM

**Using Ollama:**
```bash
# Pull a multilingual model (supports Armenian)
ollama pull qwen2.5:7b-instruct
# or
ollama pull llama3:8b-instruct

# Start Ollama (usually runs automatically)
ollama serve
```

## Usage

### Frontend

1. Upload an image with Armenian text
2. Check "Use AI Vision Model (Advanced)"
3. Select "Local (OCR + LLM) - Supports Armenian"
4. Enter languages: `en,es,hy` (or leave empty for default)
5. Enter your prompt: e.g., "Extract all Armenian text and translate to English"
6. Click "Extract with AI Prompt"

### API

```bash
curl -X POST "http://localhost:8000/api/ocr/extract-with-prompt?provider=local&prompt=Extract%20all%20Armenian%20text&languages=en,es,hy" \
  -F "file=@armenian_document.png"
```

### Python

```python
from app.services.ocr_local_vision_service import get_local_vision_service

service = get_local_vision_service()
result = await service.extract_with_prompt(
    image_data=image_bytes,
    prompt="Extract all dates, amounts, and names from this Armenian invoice",
    languages=['en', 'es', 'hy'],
    model="qwen2.5:7b-instruct"  # Optional
)

print(result['text'])  # LLM-processed result
print(result['ocr_text'])  # Raw OCR text
```

## Example Prompts for Armenian Documents

- "Extract all Armenian text from this document"
- "Translate the Armenian text to English"
- "Extract dates, amounts, and names from this Armenian invoice"
- "Find all phone numbers and addresses in Armenian"
- "Extract the table data and format as JSON"
- "What is the main topic of this Armenian document?"

## Advantages

✅ **Armenian Support**: Uses Tesseract which has excellent Armenian OCR  
✅ **Completely Offline**: No internet required  
✅ **Privacy**: All processing happens locally  
✅ **Prompt-Based**: Extract exactly what you need  
✅ **Cost-Free**: No API costs  
✅ **Multi-Language**: Supports English, Spanish, Armenian simultaneously  

## Troubleshooting

### Tesseract Not Found
```bash
# macOS
brew install tesseract tesseract-lang

# Verify
tesseract --version
tesseract --list-langs | grep hye
```

### LLM Not Responding
- Check `LLM_API_URL` in `.env` matches your LLM server
- Verify LLM is running: `curl http://localhost:11434/api/tags` (for Ollama)
- Check model name matches installed model

### Poor Armenian OCR Results
- Ensure image quality is good (300+ DPI recommended)
- Use `languages=hy` or `languages=en,hy` for Armenian-only or bilingual docs
- Preprocess images: increase contrast, convert to grayscale

## Model Recommendations

For best Armenian support with local LLMs:

1. **Qwen2.5** - Excellent multilingual support including Armenian
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```

2. **Llama 3** - Good multilingual capabilities
   ```bash
   ollama pull llama3:8b-instruct
   ```

3. **Mistral** - Strong multilingual performance
   ```bash
   ollama pull mistral:7b-instruct
   ```

## Response Format

```json
{
  "text": "LLM-processed result based on your prompt",
  "model": "qwen2.5:7b-instruct",
  "provider": "local",
  "prompt": "Your prompt",
  "ocr_text": "Raw OCR extracted text",
  "detected_languages": ["en", "hy"],
  "method": "ocr+llm"
}
```
