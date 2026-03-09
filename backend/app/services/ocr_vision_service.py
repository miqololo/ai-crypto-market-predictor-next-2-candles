"""Vision-language model service for prompt-based OCR extraction."""
import io
import base64
import logging
import os
from typing import Optional, Dict, Any
import httpx
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)


class OCRVisionService:
    """Service for prompt-based OCR using vision-language models."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def _encode_image(self, image_data: bytes) -> str:
        """Encode image bytes to base64."""
        return base64.b64encode(image_data).decode('utf-8')
    
    def _get_image_format(self, image_data: bytes) -> str:
        """Detect image format from bytes."""
        try:
            image = Image.open(io.BytesIO(image_data))
            format_map = {
                'JPEG': 'jpeg',
                'PNG': 'png',
                'GIF': 'gif',
                'WEBP': 'webp',
                'BMP': 'bmp',
                'TIFF': 'tiff'
            }
            return format_map.get(image.format, 'png')
        except Exception:
            return 'png'
    
    async def extract_with_prompt(
        self,
        image_data: bytes,
        prompt: str,
        model: Optional[str] = None,
        provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Extract information from image using vision-language model with custom prompt.
        
        Args:
            image_data: Image bytes
            prompt: Custom prompt describing what to extract
            model: Model name (optional, uses default from config)
            provider: API provider ("openai", "anthropic", "google", "ollama")
        
        Returns:
            Dictionary with extracted information and metadata
        """
        if provider == "openai":
            return await self._extract_with_openai_vision(image_data, prompt, model)
        elif provider == "anthropic":
            return await self._extract_with_anthropic(image_data, prompt, model)
        elif provider == "google":
            return await self._extract_with_google_vision(image_data, prompt, model)
        elif provider == "ollama":
            return await self._extract_with_ollama_vision(image_data, prompt, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _extract_with_openai_vision(
        self,
        image_data: bytes,
        prompt: str,
        model: Optional[str]
    ) -> Dict[str, Any]:
        """Extract using OpenAI Vision API (GPT-4 Vision, GPT-4o, etc.)."""
        api_key = self.settings.llm_api_key or os.getenv("OPENAI_API_KEY")
        api_url = self.settings.llm_api_url or "https://api.openai.com/v1"
        
        if not api_key and api_url == "https://api.openai.com/v1":
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or LLM_API_KEY in .env")
        
        model_name = model or "gpt-4o"  # Default to GPT-4o for vision
        
        image_base64 = self._encode_image(image_data)
        image_format = self._get_image_format(image_data)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = await client.post(
                f"{api_url}/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 4000
                },
                headers=headers
            )
            
            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(f"OpenAI API error: {error_text}")
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return {
                "text": content,
                "model": model_name,
                "provider": "openai",
                "prompt": prompt,
                "usage": result.get("usage", {})
            }
    
    async def _extract_with_anthropic(
        self,
        image_data: bytes,
        prompt: str,
        model: Optional[str]
    ) -> Dict[str, Any]:
        """Extract using Anthropic Claude Vision API."""
        api_key = os.getenv("ANTHROPIC_API_KEY") or self.settings.llm_api_key
        api_url = "https://api.anthropic.com/v1"
        
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY in .env")
        
        model_name = model or "claude-3-5-sonnet-20241022"
        
        image_base64 = self._encode_image(image_data)
        image_format = self._get_image_format(image_data)
        
        # Determine media type
        media_type_map = {
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        media_type = media_type_map.get(image_format, 'image/png')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            
            response = await client.post(
                f"{api_url}/messages",
                json={
                    "model": model_name,
                    "max_tokens": 4096,
                    "messages": messages
                },
                headers=headers
            )
            
            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(f"Anthropic API error: {error_text}")
            
            result = response.json()
            content = result["content"][0]["text"]
            
            return {
                "text": content,
                "model": model_name,
                "provider": "anthropic",
                "prompt": prompt,
                "usage": result.get("usage", {})
            }
    
    async def _extract_with_google_vision(
        self,
        image_data: bytes,
        prompt: str,
        model: Optional[str]
    ) -> Dict[str, Any]:
        """Extract using Google Gemini Vision API."""
        api_key = os.getenv("GOOGLE_API_KEY") or self.settings.llm_api_key
        api_url = "https://generativelanguage.googleapis.com/v1beta"
        
        if not api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY in .env")
        
        model_name = model or "gemini-1.5-pro"
        
        image_base64 = self._encode_image(image_data)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{api_url}/models/{model_name}:generateContent?key={api_key}",
                json={
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
                                },
                                {
                                    "inline_data": {
                                        "mime_type": f"image/{self._get_image_format(image_data)}",
                                        "data": image_base64
                                    }
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "maxOutputTokens": 4096
                    }
                }
            )
            
            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(f"Google API error: {error_text}")
            
            result = response.json()
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            
            return {
                "text": content,
                "model": model_name,
                "provider": "google",
                "prompt": prompt,
                "usage": result.get("usageMetadata", {})
            }
    
    async def _extract_with_ollama_vision(
        self,
        image_data: bytes,
        prompt: str,
        model: Optional[str]
    ) -> Dict[str, Any]:
        """Extract using Ollama with vision models (llava, bakllava, etc.)."""
        api_url = self.settings.llm_api_url or "http://localhost:11434"
        model_name = model or "llava:latest"
        
        image_base64 = self._encode_image(image_data)
        
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64]
            }
        ]
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {}
            if self.settings.llm_api_key:
                headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
            
            response = await client.post(
                f"{api_url}/api/chat",
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": False
                },
                headers=headers
            )
            
            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(f"Ollama API error: {error_text}")
            
            result = response.json()
            content = result["message"]["content"]
            
            return {
                "text": content,
                "model": model_name,
                "provider": "ollama",
                "prompt": prompt
            }


# Global instance
_vision_service: Optional[OCRVisionService] = None


def get_vision_service() -> OCRVisionService:
    """Get or create the global vision service instance."""
    global _vision_service
    if _vision_service is None:
        _vision_service = OCRVisionService()
    return _vision_service
