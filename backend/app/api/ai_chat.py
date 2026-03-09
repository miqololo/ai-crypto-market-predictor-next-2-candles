"""AI Chat API endpoints with SSE streaming."""
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import uuid

from app.services.ai_chat_service import get_chat_service


router = APIRouter(prefix="/ai/chat", tags=["ai-chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    strategy_context: Optional[str] = None
    strategy_id: Optional[str] = None  # If provided, update existing strategy instead of creating new


class CodeReviewRequest(BaseModel):
    code_content: str
    strategy_id: Optional[str] = None  # If provided, update strategy file after refactoring


@router.post("/stream")
async def stream_chat(request: ChatRequest):
    """
    Stream AI chat response with SSE.
    
    Returns Server-Sent Events stream with:
    - type: "status" | "content" | "complete" | "error"
    - status: "thinking" | "generating" | "complete" (for status type)
    - message: Human-readable message
    - content: Partial content chunk (for content type)
    - code: Generated Python code (for complete type)
    - params: Extracted parameters (for complete type)
    """
    session_id = request.session_id or str(uuid.uuid4())
    chat_service = get_chat_service()
    
    async def generate():
        try:
            async for chunk in chat_service.stream_chat(
                message=request.message,
                session_id=session_id,
                strategy_context=request.strategy_context,
                strategy_id=request.strategy_id
            ):
                # Format as SSE
                data = json.dumps(chunk)
                yield f"data: {data}\n\n"
            
            # Send session ID at the end
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            
        except Exception as e:
            error_data = json.dumps({
                "type": "error",
                "message": f"Stream error: {str(e)}"
            })
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/clear")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session."""
    chat_service = get_chat_service()
    chat_service.clear_conversation(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session."""
    chat_service = get_chat_service()
    history = chat_service.get_conversation(session_id)
    return {"session_id": session_id, "messages": history}


@router.post("/review/stream")
async def stream_code_review(request: CodeReviewRequest):
    """
    Review and refactor strategy code, converting static values to dynamic parameters.
    
    Returns Server-Sent Events stream with:
    - type: "status" | "content" | "complete" | "error"
    - status: "reviewing" | "extracting" | "complete"
    - message: Human-readable message
    - content: Partial content chunks
    - code: Refactored Python code (for complete type)
    - params: Extracted parameters (for complete type)
    """
    chat_service = get_chat_service()
    
    async def generate():
        try:
            async for chunk in chat_service.review_and_refactor_code(
                code_content=request.code_content,
                strategy_id=request.strategy_id
            ):
                # Format as SSE
                data = json.dumps(chunk)
                yield f"data: {data}\n\n"
            
        except Exception as e:
            error_data = json.dumps({
                "type": "error",
                "message": f"Review error: {str(e)}"
            })
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
