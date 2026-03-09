"""Run FastAPI server."""
# Avoid LightGBM/OpenMP SIGSEGV on macOS ARM64 when running under uvloop
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
