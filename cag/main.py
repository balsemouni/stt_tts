"""
CAG Microservice — Gateway-Ready Entry Point  v4.2
==================================================

FIXES v4.2
──────────
ROOT CAUSE of tokens=0:
  stream_query() is a synchronous generator that internally spawns its OWN
  thread (via Thread + TextIteratorStreamer in cag_system.py/_generate_thread).
  Wrapping it in run_in_executor() put the generator's for-loop in a thread-
  pool worker, while _generate_thread ran in yet another thread — the
  call_soon_threadsafe() inside stream_query fed tokens into the asyncio queue,
  but the thread-pool worker was BLOCKED on `for chunk in streamer` and never
  called loop.call_soon_threadsafe at all.  Result: 0 tokens, timeout every time.

  The fix: run a thin _producer() in the executor that iterates stream_query()
  synchronously and pushes each token into the asyncio queue via
  call_soon_threadsafe.  stream_query's own internal Thread handles the GPU
  work; the executor thread is just a lightweight drainer.

Other changes:
  - TOKEN_TIMEOUT_S default raised 8s → 30s.
  - gpu_lock released only after producer thread signals done.
"""

import asyncio
import collections
import hashlib
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from statistics import mean, quantiles
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from cag_config import CAGConfig, get_config_preset
from cag_system import CAGSystemFreshSession

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("cag.service")

# ── Tunables from env ─────────────────────────────────────────────────────────

DEDUP_WINDOW_S   = float(os.getenv("DEDUP_WINDOW_S",  "2.0"))
TOKEN_TIMEOUT_S  = float(os.getenv("TOKEN_TIMEOUT_S", "30.0"))
DEDUP_CACHE_SIZE = 4


# ── Metrics ───────────────────────────────────────────────────────────────────

class _Metrics:
    def __init__(self):
        self._total_queries = 0
        self._total_errors  = 0
        self._latencies_ms  : collections.deque = collections.deque(maxlen=200)
        self._lock          = threading.Lock()

    def record(self, latency_ms: float, error: bool = False):
        with self._lock:
            self._total_queries += 1
            if error:
                self._total_errors += 1
            self._latencies_ms.append(latency_ms)

    def snapshot(self) -> dict:
        with self._lock:
            lats = list(self._latencies_ms)
        avg = round(mean(lats), 1) if lats else 0.0
        p95 = round(quantiles(lats, n=20)[18], 1) if len(lats) >= 20 else avg
        return {
            "total_queries":  self._total_queries,
            "total_errors":   self._total_errors,
            "avg_latency_ms": avg,
            "p95_latency_ms": p95,
        }


metrics = _Metrics()


# ── Dedup guard ───────────────────────────────────────────────────────────────

class _DedupGuard:
    def __init__(self, window_s: float = DEDUP_WINDOW_S, maxsize: int = DEDUP_CACHE_SIZE):
        self._window  = window_s
        self._maxsize = maxsize
        self._cache   : collections.OrderedDict[str, float] = collections.OrderedDict()
        self._lock    = threading.Lock()

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def is_duplicate(self, query: str) -> bool:
        key = self._key(query)
        now = time.monotonic()
        with self._lock:
            if key in self._cache:
                if now - self._cache[key] < self._window:
                    return True
            self._cache[key] = now
            self._cache.move_to_end(key)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
        return False

    def clear(self):
        with self._lock:
            self._cache.clear()


dedup = _DedupGuard()


# ── Service state ─────────────────────────────────────────────────────────────

class ServiceState:
    def __init__(self):
        self.config    : Optional[CAGConfig]             = None
        self.cag       : Optional[CAGSystemFreshSession] = None
        self.ready     = False
        self.boot_time : Optional[datetime]              = None
        self._gpu_lock = asyncio.Lock()

    async def startup(self):
        log.info("=== CAG SERVICE STARTUP ===")
        preset = os.getenv("CAG_PRESET", "").strip()
        self.config = get_config_preset(preset) if preset else CAGConfig.from_env()
        _validate_config(self.config)
        self.cag       = CAGSystemFreshSession(self.config)
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
        if self.cag is None:
            return
        try:
            self.cag._fast_reset()
        except Exception as e:
            log.warning(f"Fast reset failed ({e}), trying full reset")
            try:
                self.cag.reset_conversation()
            except Exception as e2:
                log.error(f"Full reset also failed: {e2}")


def _validate_config(cfg: CAGConfig):
    errors: List[str] = []
    if getattr(cfg, "max_new_tokens", 1) < 1:
        errors.append("max_new_tokens must be >= 1")
    if getattr(cfg, "max_context_tokens", 1) < 64:
        errors.append("max_context_tokens must be >= 64")
    if errors:
        raise RuntimeError("CAGConfig validation failed: " + "; ".join(errors))


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
    version="4.2.0",
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
    reset_session: bool = Field(default=True)
    turn_id: Optional[str] = Field(default=None)


class ChatResponse(BaseModel):
    answer       : str
    user_name    : Optional[str] = None
    query_number : int
    turn_id      : str
    success      : bool


class HealthResponse(BaseModel):
    status         : str
    uptime_seconds : Optional[float]
    gpu_free_mb    : Optional[int]
    gpu_temp_c     : Optional[int]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_ready():
    if not svc.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading. Try again shortly.",
        )


def _make_turn_id(client_id: Optional[str]) -> str:
    return client_id or str(uuid.uuid4())


def _gpu_temp() -> Optional[int]:
    try:
        if torch.cuda.is_available():
            return torch.cuda.temperature()
    except Exception:
        pass
    return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    gpu_free = None
    if torch.cuda.is_available():
        gpu_free = torch.cuda.mem_get_info()[0] // 1024 ** 2
    uptime = None
    if svc.boot_time:
        uptime = (datetime.utcnow() - svc.boot_time).total_seconds()
    return HealthResponse(
        status="ok",
        uptime_seconds=uptime,
        gpu_free_mb=gpu_free,
        gpu_temp_c=_gpu_temp(),
    )


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
            "temp_c":          _gpu_temp(),
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


@app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
async def prometheus_metrics():
    snap = metrics.snapshot()
    lines = [
        "# HELP cag_total_queries Total number of inference requests",
        "# TYPE cag_total_queries counter",
        f"cag_total_queries {snap['total_queries']}",
        "# HELP cag_total_errors Total number of failed requests",
        "# TYPE cag_total_errors counter",
        f"cag_total_errors {snap['total_errors']}",
        "# HELP cag_avg_latency_ms Average inference latency (ms, last 200 samples)",
        "# TYPE cag_avg_latency_ms gauge",
        f"cag_avg_latency_ms {snap['avg_latency_ms']}",
        "# HELP cag_p95_latency_ms P95 inference latency (ms, last 200 samples)",
        "# TYPE cag_p95_latency_ms gauge",
        f"cag_p95_latency_ms {snap['p95_latency_ms']}",
    ]
    return "\n".join(lines) + "\n"


@app.post("/reset", tags=["chat"])
async def reset_session():
    _assert_ready()
    async with svc._gpu_lock:
        await asyncio.get_event_loop().run_in_executor(None, svc.reset_session)
    dedup.clear()
    log.info("CAG session reset via /reset")
    return {"reset": True}


# ── Chat (batch) ──────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest):
    _assert_ready()
    turn_id = _make_turn_id(req.turn_id)

    if dedup.is_duplicate(req.message):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "duplicate_query", "turn_id": turn_id},
        )

    t0         = time.monotonic()
    error_flag = False
    try:
        async with svc._gpu_lock:
            if req.reset_session:
                await asyncio.get_event_loop().run_in_executor(None, svc.reset_session)
            result = await asyncio.get_event_loop().run_in_executor(
                None, svc.cag.query, req.message
            )

        if not result.get("success"):
            error_flag = True
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Inference error"),
            )

        log.info(
            f"[turn:{turn_id}] chat OK "
            f"q={len(req.message)}ch "
            f"lat={round((time.monotonic()-t0)*1000)}ms"
        )
        return ChatResponse(
            answer       = result["answer"],
            user_name    = result.get("user_name"),
            query_number = result["query_number"],
            turn_id      = turn_id,
            success      = True,
        )
    finally:
        metrics.record((time.monotonic() - t0) * 1000, error=error_flag)


# ── Chat (streaming SSE) ──────────────────────────────────────────────────────

@app.post("/chat/stream", tags=["chat"])
async def chat_stream(req: ChatRequest):
    """
    Single-turn streaming chat via Server-Sent Events.

    Token format:    data: <token>\\n\\n
    Turn header:     data: [TURN_ID] <uuid>\\n\\n   ← first event
    Done signal:     data: [DONE]\\n\\n
    Error signal:    data: [ERROR] <message>\\n\\n
    Timeout signal:  data: [TIMEOUT]\\n\\n

    KEY DESIGN NOTE
    ───────────────
    stream_query() is a sync generator that internally spawns its own Thread
    (via TextIteratorStreamer + _generate_thread in cag_system.py).

    We run a thin _producer() in run_in_executor — it iterates stream_query()
    synchronously and forwards each token to the asyncio queue via
    call_soon_threadsafe.  The GPU work stays on stream_query's daemon Thread.

    Do NOT call stream_query() inside an async for loop directly — it would
    block the event loop on the TextIteratorStreamer.
    """
    _assert_ready()
    turn_id = _make_turn_id(req.turn_id)

    if dedup.is_duplicate(req.message):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"error": "duplicate_query", "turn_id": turn_id},
        )

    async def _event_generator() -> AsyncGenerator[str, None]:
        loop         = asyncio.get_event_loop()
        q            : asyncio.Queue = asyncio.Queue()
        cancel_event = threading.Event()
        t0           = time.monotonic()
        token_count  = [0]
        error_flag   = [False]

        # ── Turn-id header (gateway correlation) ──────────────────────────
        yield f"data: [TURN_ID] {turn_id}\n\n"

        def _producer():
            """
            Thread-pool worker: drains stream_query() and forwards tokens.
            stream_query() manages its own GPU thread internally.
            """
            try:
                if req.reset_session:
                    svc.reset_session()

                for chunk in svc.cag.stream_query(req.message):
                    if cancel_event.is_set():
                        log.info(f"[turn:{turn_id}] stream cancelled by client")
                        break
                    if chunk:
                        token_count[0] += 1
                        loop.call_soon_threadsafe(q.put_nowait, ("token", chunk))

            except Exception as exc:
                error_flag[0] = True
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))

        try:
            async with svc._gpu_lock:
                producer_future = loop.run_in_executor(None, _producer)

                try:
                    while True:
                        try:
                            kind, value = await asyncio.wait_for(
                                q.get(), timeout=TOKEN_TIMEOUT_S
                            )
                        except asyncio.TimeoutError:
                            log.warning(
                                f"[turn:{turn_id}] token timeout after {TOKEN_TIMEOUT_S}s"
                            )
                            cancel_event.set()
                            error_flag[0] = True
                            yield f"data: [TIMEOUT]\n\n"
                            break

                        if kind == "token":
                            yield f"data: {value}\n\n"
                        elif kind == "error":
                            log.error(f"[turn:{turn_id}] producer error: {value}")
                            yield f"data: [ERROR] {value}\n\n"
                            break
                        else:  # "done"
                            break

                finally:
                    # Always signal stop and wait for the producer thread to
                    # finish before releasing the GPU lock.
                    cancel_event.set()
                    try:
                        await asyncio.wait_for(
                            asyncio.wrap_future(producer_future), timeout=10.0
                        )
                    except Exception:
                        pass

        except asyncio.CancelledError:
            cancel_event.set()
            log.info(f"[turn:{turn_id}] stream cancelled (client disconnect)")
            return

        finally:
            lat_ms = (time.monotonic() - t0) * 1000
            metrics.record(lat_ms, error=error_flag[0])
            log.info(
                f"[turn:{turn_id}] stream done "
                f"tokens={token_count[0]} "
                f"lat={round(lat_ms)}ms"
            )

        yield f"data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Turn-Id":         turn_id,
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