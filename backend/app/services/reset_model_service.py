"""
Reset AI trained models: llm1_collection, ChromaDB, FAISS pattern index.
"""
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from pymongo import MongoClient

from app.services.llm1_service import DB_NAME, LLM1_COLLECTION, LLM1_CANDLES_COLLECTION, MONGODB_URI
from app.services.llm1_train_service import LLM1_INDEX_BASE

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
PATTERNS_BASE_DIR = Path("data/patterns")


def reset_llm1_index() -> Dict[str, Any]:
    """Delete trained LLM1 FAISS index files."""
    base = LLM1_INDEX_BASE if LLM1_INDEX_BASE.is_absolute() else Path.cwd() / LLM1_INDEX_BASE
    removed = []
    if base.exists():
        for item in base.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                removed.append(item.name)
    return {"llm1_index": {"deleted": bool(removed), "removed": removed}}


def reset_llm1_collection(mongodb_uri: str = None) -> Dict[str, Any]:
    """Drop all documents from llm1_collection, llm1_candles, and trained index."""
    uri = mongodb_uri or MONGODB_URI
    client = MongoClient(uri)
    db = client[DB_NAME]
    meta_deleted = db[LLM1_COLLECTION].delete_many({}).deleted_count
    candles_deleted = db[LLM1_CANDLES_COLLECTION].delete_many({}).deleted_count
    index_result = reset_llm1_index()
    return {
        "llm1_collection": {"deleted": meta_deleted},
        "llm1_candles": {"deleted": candles_deleted},
        "llm1_index": index_result["llm1_index"],
    }


def reset_chroma_db(persist_dir: str = None) -> Dict[str, Any]:
    """Delete ChromaDB persist directory if it exists."""
    path = Path(persist_dir or CHROMA_PERSIST_DIR)
    if path.exists():
        shutil.rmtree(path)
        return {"chroma_db": {"deleted": True, "path": str(path)}}
    return {"chroma_db": {"deleted": False, "path": str(path), "reason": "not found"}}


def reset_faiss_patterns(base_dir: Path = None) -> Dict[str, Any]:
    """Delete all FAISS pattern index files in data/patterns/."""
    base = base_dir or PATTERNS_BASE_DIR
    if not base.is_absolute():
        base = Path.cwd() / base
    deleted_dirs: List[str] = []
    if base.exists():
        for item in base.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                deleted_dirs.append(item.name)
        if deleted_dirs:
            return {"faiss_patterns": {"deleted": True, "removed": deleted_dirs}}
    return {"faiss_patterns": {"deleted": True if deleted_dirs else False, "removed": deleted_dirs}}


def reset_all_models(
    llm1: bool = True,
    chroma: bool = True,
    faiss: bool = True,
    mongodb_uri: str = None,
) -> Dict[str, Any]:
    """
    Reset AI trained models. Returns summary of what was cleared.
    """
    result: Dict[str, Any] = {"success": True, "reset": {}}
    try:
        if llm1:
            result["reset"]["llm1"] = reset_llm1_collection(mongodb_uri)
        if chroma:
            result["reset"]["chroma"] = reset_chroma_db()
        if faiss:
            result["reset"]["faiss"] = reset_faiss_patterns()
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
    return result
