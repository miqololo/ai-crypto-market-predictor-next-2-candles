"""API routes."""
from fastapi import APIRouter
from .llm1 import router as llm1_router
from .routes import router as routes_router
from .ai_similarity import router as ai_similarity_router
from .ai_chat import router as ai_chat_router
from .ocr import router as ocr_router
from .rf_ml import router as rf_ml_router

router = APIRouter()
router.include_router(routes_router)
router.include_router(ai_similarity_router)
router.include_router(llm1_router, prefix="/llm1", tags=["llm1"])
router.include_router(rf_ml_router, prefix="/rf", tags=["rf"])
router.include_router(ai_chat_router)
router.include_router(ocr_router)
