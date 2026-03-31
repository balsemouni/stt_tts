"""
azure_tts.py — Azure Cognitive Services TTS REST client with prosody
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from xml.sax.saxutils import escape

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import httpx

# ─── Configuration ────────────────────────────────────────────────────────────

AZURE_TTS_KEY      = os.getenv("AZURE_TTS_KEY",      "")
AZURE_TTS_ENDPOINT = os.getenv("AZURE_TTS_ENDPOINT", "https://francecentral.tts.speech.microsoft.com/cognitiveservices/v1")
AZURE_TTS_VOICE    = os.getenv("AZURE_TTS_VOICE",    "en-US-AriaNeural")
AZURE_TTS_FORMAT   = "raw-24khz-16bit-mono-pcm"


# ─── Prosody detection ────────────────────────────────────────────────────────

def _detect_tone(text: str) -> str:
    """Detect the emotional tone of a phrase for SSML style/prosody."""
    s = text.strip()
    low = s.lower()
    # Greetings — cheerful
    _GREET = re.compile(r"\b(hello|hi|hey|welcome|glad|great to)\b", re.I)
    if _GREET.search(s) or s.endswith("!"):
        return "cheerful"
    # Questions — friendly, inquisitive
    if s.endswith("?"):
        return "question"
    # Empathy phrases
    _EMPATHY = re.compile(
        r"\b(sorry|understand|sounds like|that must|i hear you|no worries|"
        r"i see|that's tough|frustrating|difficult)\b", re.I
    )
    if _EMPATHY.search(s):
        return "empathetic"
    # Short confirmations — gentle/chat
    if len(s.split()) <= 4:
        return "calm"
    # Default — calm, conversational
    return "calm"


# ─── SSML builder with prosody ────────────────────────────────────────────────

def build_ssml(text: str, tone: str | None = None) -> str:
    """
    Build Azure SSML with natural prosody based on phrase tone.
    Uses <mstts:express-as> for AriaNeural style and <prosody> for fine-tuning.
    """
    safe = escape(text.strip())
    if not tone:
        tone = _detect_tone(text)

    # AriaNeural supports styles: chat, cheerful, empathetic, friendly, etc.
    if tone == "question":
        # Slightly higher pitch, medium rate — sounds inquisitive
        inner = (
            f'<mstts:express-as style="friendly">'
            f'<prosody rate="0%" pitch="+5%">{safe}</prosody>'
            f'</mstts:express-as>'
        )
    elif tone == "cheerful":
        # Upbeat for greetings/exclamations
        inner = (
            f'<mstts:express-as style="cheerful">'
            f'<prosody rate="+5%" pitch="+3%">{safe}</prosody>'
            f'</mstts:express-as>'
        )
    elif tone == "empathetic":
        # Slower, softer for empathy
        inner = (
            f'<mstts:express-as style="empathetic">'
            f'<prosody rate="-5%" pitch="-2%">{safe}</prosody>'
            f'</mstts:express-as>'
        )
    else:
        # Calm/conversational — chat style, natural pace
        inner = (
            f'<mstts:express-as style="chat">'
            f'<prosody rate="0%" pitch="0%">{safe}</prosody>'
            f'</mstts:express-as>'
        )

    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">'
        f'<voice name="{AZURE_TTS_VOICE}">'
        f'{inner}'
        '</voice></speak>'
    )


# ─── Persistent HTTP client (reuses TCP+TLS connections) ──────────────────────

import logging as _logging
_log = _logging.getLogger("tts")

_pool: httpx.AsyncClient | None = None

def _get_pool() -> httpx.AsyncClient:
    global _pool
    if _pool is None or _pool.is_closed:
        _pool = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
            limits=httpx.Limits(max_connections=6, max_keepalive_connections=4),
        )
    return _pool


async def close_pool():
    """Call on shutdown to cleanly close the connection pool."""
    global _pool
    if _pool and not _pool.is_closed:
        await _pool.aclose()
        _pool = None


# ─── REST request (persistent pool — no handshake per call) ──────────────────

_HEADERS = {
    "Ocp-Apim-Subscription-Key": AZURE_TTS_KEY,
    "X-Microsoft-OutputFormat":  AZURE_TTS_FORMAT,
    "Content-Type":              "application/ssml+xml",
}

async def azure_tts_request(text: str, tone: str | None = None) -> bytes:
    """Call Azure TTS REST API and return raw PCM bytes (24kHz 16-bit mono)."""
    ssml = build_ssml(text, tone=tone)
    client = _get_pool()
    resp = await client.post(AZURE_TTS_ENDPOINT, content=ssml, headers=_HEADERS)
    resp.raise_for_status()
    return resp.content


async def azure_tts_stream(text: str, tone: str | None = None, chunk_size: int = 4096):
    """Stream PCM chunks as they arrive from Azure — yields bytes chunks."""
    ssml = build_ssml(text, tone=tone)
    client = _get_pool()
    async with client.stream("POST", AZURE_TTS_ENDPOINT,
                             content=ssml, headers=_HEADERS) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes(chunk_size):
            yield chunk
