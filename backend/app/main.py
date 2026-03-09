"""FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router as api_router

app = FastAPI(title="Traider API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router, prefix="/api", tags=["api"])


@app.get("/health")
def health():
    return {"status": "ok"}
