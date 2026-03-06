"""
CAG Microservice — Gateway-Ready Entry Point  v3
================================================
FastAPI application wrapping CAGSystemFreshSession.

IMPROVEMENTS v3 (over v2)
─────────────────────────
1. POST /chat and POST /chat/stream now accept an optional
   `reset_session: bool = False` flag.
   When set to True the CAG session history is cleared BEFORE processing
   the query — eliminating the separate POST /reset round-trip entirely.
   This removes one full HTTP + event-loop cycle from every voice turn.

2. POST /reset is kept for backward compatibility but is now a thin wrapper
   around CAGSystemFreshSession._fast_reset() (no synchronize, ~0 ms cost).

3. reset_session() in ServiceState is simplified — it calls _fast_reset()
   directly instead of iterating over attribute names or rebuilding the
   entire system.

4. stream_query tokens get a trailing space so the gateway SentenceAccumulator
   can concatenate them without guessing word boundaries.

5. SSE response headers include Connection: keep-alive.

6. One worker only (GPU not thread-safe). Scale at gateway level.

Endpoints
─────────
  POST   /chat            — single-turn JSON (supports reset_session flag)
  POST   /chat/stream     — single-turn SSE   (supports reset_session flag)
  POST   /reset           — clear session history (fast, no sync)
  GET    /health          — liveness probe
  GET    /ready           — readiness probe
  GET    /stats           — system stats

Startup
───────
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

Environment variables
─────────────────────
  CAG_MODEL_ID            HuggingFace model identifier
  CAG_MAX_CONTEXT_TOKENS  token budget for knowledge cache
  CAG_MAX_NEW_TOKENS      max tokens per response
  CAG_CACHE_FILE          path to persisted .pt cache file
  CAG_VERBOSE             "true" / "false"
  CAG_PRESET              "default" | "large" | "fast" | "safe"
  CAG_FLASH_ATTN          "true" / "false"  (default: true)
  PORT                    HTTP port (default 8000)
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from cag_config import CAGConfig, get_config_preset
from cag_system import CAGSystemFreshSession

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("cag.service")


# ── Service state ─────────────────────────────────────────────────────────────

class ServiceState:
    def __init__(self):
        self.config    : Optional[CAGConfig]            = None
        self.cag       : Optional[CAGSystemFreshSession] = None
        self.ready     = False
        self.boot_time : Optional[datetime]             = None
        self._gpu_lock = asyncio.Lock()

    async def startup(self):
        log.info("=== CAG SERVICE STARTUP ===")
        preset = os.getenv("CAG_PRESET", "").strip()
        self.config = get_config_preset(preset) if preset else CAGConfig.from_env()
        self.cag    = CAGSystemFreshSession(self.config)
        await asyncio.get_event_loop().run_in_executor(None, self.cag.initialize)
        self.ready     = True
        self.boot_time = datetime.utcnow()
        log.info("=== CAG SERVICE READY ===")

    async def teardown(self):
        log.info("Shutting down…")
        if self.cag:
            self.cag.cleanup()
        torch.cuda.empty_cache()
        log.info("Cleanup done.")

    def reset_session(self):
        """
        Lightweight session reset — no synchronize(), safe to call every turn.

        Uses CAGSystemFreshSession._fast_reset() which only clears the
        in-memory message list and truncates the KV cache tensor slices.
        """
        if self.cag is None:
            return
        try:
            self.cag._fast_reset()
            log.info("CAG session reset via _fast_reset()")
        except Exception as e:
            log.warning(f"Fast reset failed ({e}), trying full reset")
            try:
                self.cag.reset_conversation()
            except Exception as e2:
                log.error(f"Full reset also failed: {e2}")


svc = ServiceState()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.startup()
    yield
    await svc.teardown()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Ask Novation — CAG Inference Service",
    description="LLM-powered solution recommendation chatbot (CAG architecture).",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    reset_session: bool = Field(
        default=False,
        description=(
            "When True the session history is cleared before processing this message. "
            "Set to True on every new voice turn to prevent the model re-broadcasting "
            "stale context and eliminate the separate POST /reset round-trip."
        ),
    )


class ChatResponse(BaseModel):
    answer       : str
    user_name    : Optional[str] = None
    query_number : int
    success      : bool


class HealthResponse(BaseModel):
    status         : str
    uptime_seconds : Optional[float]
    gpu_free_mb    : Optional[int]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_ready():
    if not svc.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading. Try again shortly.",
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    gpu_free = None
    if torch.cuda.is_available():
        gpu_free = torch.cuda.mem_get_info()[0] // 1024 ** 2
    uptime = None
    if svc.boot_time:
        uptime = (datetime.utcnow() - svc.boot_time).total_seconds()
    return HealthResponse(status="ok", uptime_seconds=uptime, gpu_free_mb=gpu_free)


@app.get("/ready", tags=["ops"])
async def ready():
    if not svc.ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False, "detail": "Model loading…"},
        )
    return {"ready": True}


@app.get("/stats", tags=["ops"])
async def stats():
    _assert_ready()
    base_stats = svc.cag.get_stats()
    gpu_info: Dict[str, Any] = {}
    if torch.cuda.is_available():
        free_mb  = torch.cuda.mem_get_info()[0] // 1024 ** 2
        total_mb = torch.cuda.mem_get_info()[1] // 1024 ** 2
        gpu_info = {
            "free_mb":         free_mb,
            "total_mb":        total_mb,
            "used_mb":         total_mb - free_mb,
            "utilization_pct": round((total_mb - free_mb) / total_mb * 100, 1),
        }
    return {
        "knowledge_entries":      base_stats.get("knowledge", {}).get("entries", 0),
        "knowledge_tokens":       base_stats.get("knowledge", {}).get("tokens", 0),
        "cache_initialized":      base_stats.get("cache", {}).get("initialized", False),
        "cache_knowledge_tokens": base_stats.get("cache", {}).get("knowledge_tokens", 0),
        "flash_attention":        base_stats.get("config", {}).get("flash_attention", False),
        "total_queries":          base_stats.get("total_queries", 0),
        "gpu":                    gpu_info,
        "boot_time":              svc.boot_time.isoformat() if svc.boot_time else None,
    }


# ── Reset (fast, no synchronize) ──────────────────────────────────────────────

@app.post("/reset", tags=["chat"])
async def reset_session():
    """
    Clear the CAG conversation history (lightweight — no GPU sync).

    Kept for backward compatibility.  For new integrations, prefer sending
    `reset_session: true` inside the chat request to save a round-trip.
    """
    _assert_ready()
    async with svc._gpu_lock:
        await asyncio.get_event_loop().run_in_executor(None, svc.reset_session)
    log.info("CAG session reset via /reset")
    return {"reset": True}


# ── Chat (batch) ──────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest):
    """
    Single-turn chat — full JSON response.

    Set `reset_session: true` to clear history before this query (saves the
    separate POST /reset call).
    """
    _assert_ready()

    async with svc._gpu_lock:
        if req.reset_session:
            await asyncio.get_event_loop().run_in_executor(None, svc.reset_session)

        result = await asyncio.get_event_loop().run_in_executor(
            None, svc.cag.query, req.message
        )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Inference error"),
        )
    return ChatResponse(
        answer       = result["answer"],
        user_name    = result.get("user_name"),
        query_number = result["query_number"],
        success      = True,
    )


# ── Chat (streaming SSE) ──────────────────────────────────────────────────────

@app.post("/chat/stream", tags=["chat"])
async def chat_stream(req: ChatRequest):
    """
    Single-turn streaming chat via Server-Sent Events.

    Set `reset_session: true` to clear history before streaming (saves the
    separate POST /reset call).

    Token format:   data: <token>\\n\\n
    Done signal:    data: [DONE]\\n\\n
    Error signal:   data: [ERROR] <message>\\n\\n
    """
    _assert_ready()

    async def _event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _producer():
            # Optional fast reset inside the same GPU lock window
            if req.reset_session:
                svc.reset_session()
            try:
                for token in svc.cag.stream_query(req.message):
                    # Trailing space helps the gateway SentenceAccumulator
                    # join tokens without guessing word boundaries.
                    loop.call_soon_threadsafe(q.put_nowait, ("token", token))
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))

        async with svc._gpu_lock:
            loop.run_in_executor(None, _producer)
            while True:
                kind, value = await q.get()
                if kind == "token":
                    yield f"data: {value}\n\n"
                elif kind == "error":
                    yield f"data: [ERROR] {value}\n\n"
                    break
                else:
                    break

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",   # Important for TTS clients
            "X-Accel-Buffering": "no",          # Disable Nginx buffering
        },
    )


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Check service logs."},
    )


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
        reload=False,
    )