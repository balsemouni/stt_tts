"""
gateway.py — Zero-Latency Voice Pipeline Gateway  v14 (thin orchestrator)
═════════════════════════════════════════════════════════════════════════

ARCHITECTURE
────────────
  Client mic audio ──► STT microservice  (WebSocket, persistent)
                            │
                     word / segment events
                            ▼
                       word_buf ──► silence ──► CAG microservice (WebSocket, streaming)
                                                      │
                                               token stream → TonalAccumulator
                                                      │
                                         sentence chunks ──► Azure TTS (REST, parallel)
                                                                   │
                                                              PCM frames
                                                                   │
                                                          play_worker ──► client

Modules
───────
  gateway.session   — GatewaySession (full voice pipeline per client)
  gateway.models    — State enum, RepetitionGuard, drain_q, ws_connect
  gateway.echo_gate — TimingEchoGate, AITextEchoFilter
  gateway.latency   — LatencyTracker, TurnLatency
  gateway.tonal     — TonalAccumulator, TonalChunk, classify_tone
  tts.azure_tts     — azure_tts_request, build_ssml
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gateway.session import GatewaySession, TEST_MODE

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitoring.metrics import instrument_app

# ─── Configuration ────────────────────────────────────────────────────────────

GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8090"))
USER_SERVICE_URL    = os.getenv("USER_SERVICE_URL",    "http://localhost:8006")
SESSION_SERVICE_URL = os.getenv("SESSION_SERVICE_URL", "http://localhost:8005")
MESSAGE_SERVICE_URL = os.getenv("MESSAGE_SERVICE_URL", "http://localhost:8003")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("gateway")

app = FastAPI(title="Voice Gateway", version="14.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
instrument_app(app, service_name="gateway", version="14.0.0")

# ── Custom gateway metrics ────────────────────────────────────────────────────
from monitoring.metrics import _safe_metric
from prometheus_client import Counter, Gauge, Histogram, REGISTRY

GW_ACTIVE_SESSIONS = _safe_metric(Gauge, "gateway_active_sessions", "Number of active WebSocket sessions", REGISTRY)
GW_TOTAL_SESSIONS = _safe_metric(Counter, "gateway_total_sessions", "Total WebSocket sessions created", REGISTRY)
GW_BARGE_INS = _safe_metric(Counter, "gateway_barge_ins_total", "Total barge-in events", REGISTRY)
GW_TTS_CHUNKS = _safe_metric(Counter, "gateway_tts_chunks_total", "Total TTS chunks synthesised", REGISTRY)
GW_STT_SEGMENTS = _safe_metric(Counter, "gateway_stt_segments_total", "Total STT segments received", REGISTRY)
GW_CAG_QUERIES = _safe_metric(Counter, "gateway_cag_queries_total", "Total CAG queries sent", REGISTRY)
GW_E2E_LATENCY = _safe_metric(
    Histogram, "gateway_e2e_latency_seconds", "End-to-end turn latency", REGISTRY,
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
)
GW_STT_LATENCY = _safe_metric(
    Histogram, "gateway_stt_latency_seconds", "STT segment latency", REGISTRY,
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
GW_CAG_FIRST_TOKEN = _safe_metric(
    Histogram, "gateway_cag_first_token_seconds", "Time to first CAG token", REGISTRY,
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
GW_TTS_SYNTH_LATENCY = _safe_metric(
    Histogram, "gateway_tts_synth_seconds", "TTS synthesis latency per chunk", REGISTRY,
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

_session_latency_store: dict[str, dict] = {}


# ─── App startup ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _gateway_startup():
    log.info("[startup] TimingEchoGate ready — no fingerprint file needed.")


# ─── Auth + Session helpers ───────────────────────────────────────────────────

async def _verify_token(token: str) -> dict | None:
    """Call user_auth /auth/verify-token and return user info, or None."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.post(
                f"{USER_SERVICE_URL}/auth/verify-token",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 200:
                return resp.json()
        except httpx.RequestError:
            pass
    return None


async def _get_or_create_session(user_id: str, session_id: str | None) -> str | None:
    """Resume existing session or create a new one via session_chat service."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        if session_id:
            try:
                resp = await client.get(f"{SESSION_SERVICE_URL}/sessions/{session_id}")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("user_id") == user_id:
                        return session_id
            except httpx.RequestError:
                pass

        try:
            resp = await client.post(
                f"{SESSION_SERVICE_URL}/sessions",
                json={"user_id": user_id},
            )
            if resp.status_code == 201:
                return resp.json().get("id")
        except httpx.RequestError:
            pass
    return None


async def _save_message(session_id: str, role: str, content: str):
    """Fire-and-forget: save a message to the message service."""
    if not content.strip():
        return
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            await client.post(
                f"{MESSAGE_SERVICE_URL}/sessions/{session_id}/messages",
                json={"role": role, "content": content},
            )
        except httpx.RequestError as e:
            log.warning(f"Failed to save message: {e}")


async def _update_session_title(session_id: str, title: str):
    """Set the session title (called once from the first user message)."""
    short = title[:80].strip()
    if not short:
        return
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            await client.patch(
                f"{SESSION_SERVICE_URL}/sessions/{session_id}/title",
                json={"title": short},
            )
        except httpx.RequestError:
            pass


async def _load_history(session_id: str) -> list[dict]:
    """Fetch past messages for a session from the message service."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(
                f"{MESSAGE_SERVICE_URL}/sessions/{session_id}/messages",
                params={"limit": 50},
            )
            if r.status_code == 200:
                return r.json()
        except httpx.RequestError:
            pass
    return []


# ─── WebSocket endpoint ──────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(
    ws: WebSocket,
    token: str = Query(default=None),
    session_id: str = Query(default=None),
):
    await ws.accept()

    # ── Authenticate ──
    user_info = None
    user_id   = None
    chat_session_id = None

    if token:
        user_info = await _verify_token(token)
        if not user_info:
            await ws.send_json({"type": "error", "message": "Invalid or expired token"})
            await ws.close(code=4001, reason="Unauthorized")
            return
        user_id = user_info.get("id")
        log.info(f"Authenticated user: {user_id}")

        # ── Create or resume chat session ──
        chat_session_id = await _get_or_create_session(user_id, session_id)
        if not chat_session_id:
            await ws.send_json({"type": "error", "message": "Failed to create session"})
            await ws.close(code=4002, reason="Session error")
            return
        log.info(f"Chat session: {chat_session_id}")
    else:
        log.info("Anonymous connection (no token)")

    session  = GatewaySession(
        ws,
        chat_session_id=chat_session_id,
        save_message_fn=_save_message,
        update_title_fn=_update_session_title,
        load_history_fn=_load_history,
    )
    GW_ACTIVE_SESSIONS.inc()
    GW_TOTAL_SESSIONS.inc()
    pipeline = asyncio.create_task(session.run())
    log.info(f"[{session.sid}] client connected")

    if chat_session_id:
        await ws.send_json({"type": "session", "session_id": chat_session_id})

    if TEST_MODE:
        log.info(f"[{session.sid}] TEST_MODE — greeting skipped")
        await ws.send_json({"type": "ready", "message": "TEST MODE — ready"})

    try:
        async for raw in ws.iter_bytes():
            if not raw:
                continue
            ftype   = raw[0]
            payload = raw[1:]

            if ftype == 0x01:                           # audio frame
                session.push_audio(raw)
                session._last_pong_time = time.monotonic()

            elif ftype == 0x02:                         # control frame
                try:
                    ctrl = json.loads(payload)
                except Exception:
                    continue
                mtype = ctrl.get("type")

                if mtype == "ping":
                    await ws.send_json({"type": "pong"})

                elif mtype == "pong":
                    session._last_pong_time = time.monotonic()

                elif mtype == "inject_query":
                    text = ctrl.get("text", "").strip()
                    if text:
                        turn_id = str(uuid.uuid4())
                        session._lat.new_turn(turn_id, text)
                        session._lat.on_stt_segment()
                        await ws.send_json({"type": "segment", "text": text})
                        await session._query_q.put((turn_id, text))
                        log.info(f"[{session.sid}] inject_query: {text!r}")

                elif mtype == "get_stats":
                    await ws.send_json({
                        "type":  "stats",
                        "sid":   session.sid,
                        "state": session.state.name,
                    })

                elif mtype == "get_latency":
                    await ws.send_json({
                        "type":    "latency_snapshot",
                        "turns":   session._lat.all_reports(),
                        "summary": session._lat.session_summary(),
                    })

                elif mtype == "reset_context" and session._stt_ws:
                    try:
                        await session._stt_ws.send(b'\x02' + payload)
                    except Exception:
                        pass

    except WebSocketDisconnect:
        log.info(f"[{session.sid}] disconnected")
    except Exception as e:
        log.error(f"[{session.sid}] ws error: {e}")
    finally:
        GW_ACTIVE_SESSIONS.dec()
        await session.stop(_session_latency_store)
        pipeline.cancel()


# ─── REST ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "14.0.0"}


@app.get("/latency/session/{sid}")
def get_session_latency(sid: str):
    data = _session_latency_store.get(sid)
    if not data:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return data


@app.get("/latency/sessions")
def list_sessions():
    return {"sessions": list(_session_latency_store.keys())}


if __name__ == "__main__":
    uvicorn.run(
        "gateway.gateway:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        workers=1,
        log_level="info",
        reload=False,
    )
