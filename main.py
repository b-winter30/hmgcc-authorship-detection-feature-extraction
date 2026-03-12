"""
FastAPI application entry-point.

Creates the app, attaches middleware, registers startup hooks,
and mounts the endpoint router.
"""

import uuid
import time
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

import dependencies as deps
from endpoints import router

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_activity.log"),
    ],
)
logger = logging.getLogger("stylometry_api")


# ── Middleware ─────────────────────────────────────────────────────────────────

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.time()
        logger.info(f"ID:{request_id} | Req: {request.method} {request.url.path}")
        try:
            response = await call_next(request)
            ms = (time.time() - start_time) * 1000
            logger.info(f"ID:{request_id} | Status: {response.status_code} | Time: {ms:.2f}ms")
            return response
        except Exception as e:
            logger.error(f"ID:{request_id} | Request failed: {e}")
            raise


# ── App creation ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stylometric Feature Extraction & Authorship Detection API",
    description="Extract multilingual stylometric features and predict authorship (Pure Stylometry)",
    version="3.0.0",
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://your-nextjs-app.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load ML models and heavy resources when the server starts."""
    deps.load_models("models")
    deps.initialize_extractor_with_ntp()
    deps.initialize_tonal_classifier()


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)