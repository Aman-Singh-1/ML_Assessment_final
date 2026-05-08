"""
Dream Reflection Media – Mini ML System
Semantic Search & Retrieval API
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from app.api.routes import router
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Semantic Search API",
    description="Stores user inputs, generates embeddings, retrieves similar results",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed}ms)")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
