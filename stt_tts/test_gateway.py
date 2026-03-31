"""
test_gateway.py — Gateway Test Client v5.0  (full auth + session + messages)
══════════════════════════════════════════════════════════════════════════════

Full flow
─────────
  1. Register / Login  →  JWT token
  2. Create or resume a chat session
  3. Connect WS with ?token=…&session_id=…
  4. Stream mic audio  →  STT transcribes  →  saved as role=user message
  5. CAG responds       →  saved as role=agent message  →  TTS plays back
  6. On Ctrl+C  →  fetch & display full conversation history from DB

Architecture
────────────
  test_gateway.py  ←WS→  gateway (port 8090)
                              ├── user_auth   (8006) — JWT login
                              ├── session_chat(8005) — session CRUD
                              ├── messages    (8003) — message persistence
                              ├── STT         (8001) — speech-to-text
                              ├── CAG         (8000) — AI generation
                              └── Azure TTS          — text-to-speech

Usage
─────
    python test_gateway.py --email you@example.com --password YourPass1
    python test_gateway.py --email you@example.com --password YourPass1 --session SESSION_ID
    python test_gateway.py --register --email new@example.com --username myname --password Pass123
    python test_gateway.py                          # anonymous (no auth, no persistence)

Requirements
────────────
    pip install websockets sounddevice numpy httpx
"""

import argparse
import asyncio
import json
import logging
import sys
import datetime
import textwrap
import os
import threading
import queue as stdlib_queue
from typing import Optional

_missing = []
try:    import sounddevice as sd
except ImportError: _missing.append("sounddevice")
try:    import websockets
except ImportError: _missing.append("websockets")
try:    import numpy as np
except ImportError: _missing.append("numpy")
try:    import httpx
except ImportError: _missing.append("httpx")
if _missing:
    print(f"Missing: pip install {' '.join(_missing)}")
    sys.exit(1)

log = logging.getLogger("test_gateway")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--host",      default="localhost")
parser.add_argument("--port",      default=8090,  type=int)
parser.add_argument("--mic-rate",  default=16000, type=int)
parser.add_argument("--chunk-ms",  default=20,    type=int)
parser.add_argument("--out-rate",  default=24000, type=int)
parser.add_argument("--debug-raw", action="store_true")
# Auth & session
parser.add_argument("--email",     default=None,  help="Login email")
parser.add_argument("--password",  default=None,  help="Login password")
parser.add_argument("--username",  default=None,  help="Username (only for --register)")
parser.add_argument("--register",  action="store_true", help="Register a new account first")
parser.add_argument("--session",   default=None,  help="Resume existing session ID")
parser.add_argument("--auth-port", default=8006,  type=int)
parser.add_argument("--sess-port", default=8005,  type=int)
parser.add_argument("--msg-port",  default=8003,  type=int)
args = parser.parse_args()

GW_URL     = f"ws://{args.host}:{args.port}/ws"
AUTH_URL   = f"http://{args.host}:{args.auth_port}"
SESS_URL   = f"http://{args.host}:{args.sess_port}"
MSG_URL    = f"http://{args.host}:{args.msg_port}"
MIC_RATE   = args.mic_rate
CHUNK_MS   = args.chunk_ms
CHUNK_SAMP = int(MIC_RATE * CHUNK_MS / 1000)
OUT_RATE   = args.out_rate
DEBUG_RAW  = args.debug_raw

# ── Terminal width ────────────────────────────────────────────────────────────
try:    TW = min(os.get_terminal_size().columns, 120)
except: TW = 100

# ── ANSI colours ──────────────────────────────────────────────────────────────
R    = "\033[0m";  BOLD = "\033[1m";  DIM  = "\033[2m"
RED  = "\033[91m"; GRN  = "\033[92m"; YLW  = "\033[93m"
BLU  = "\033[94m"; MAG  = "\033[95m"; CYN  = "\033[96m"
WHT  = "\033[97m"; BLK  = "\033[30m"
BGDARK = "\033[48;5;234m"; BGCYN = "\033[46m"; BGMAG = "\033[45m"

def c(*a):
    return "".join(a[:-1]) + a[-1] + R

def ts():
    n = datetime.datetime.now()
    return c(DIM, f"{n.strftime('%H:%M:%S')}.{n.microsecond//1000:03d}")

SEP      = c(DIM, "─" * TW)
SEP_BOLD = c(CYN, "═" * TW)

# ── Global state ──────────────────────────────────────────────────────────────
_audio_in_q:  asyncio.Queue          # mic  → gateway  (bytes)
_pcm_play_q:  stdlib_queue.Queue     # gateway → speaker thread (np.ndarray float32)
_loop         = None
_stop         = asyncio.Event()

_cur_user_words:   list = []
_cur_ai_tokens:    list = []
_cur_ai_sentences: list = []
_ai_speaking       = False

# Auth state (populated at startup)
_jwt_token:     Optional[str] = None
_user_id:       Optional[str] = None
_session_id:    Optional[str] = None  # chat session for message persistence

# ── Waveform helper ───────────────────────────────────────────────────────────

def _waveform_line(pcm_f32: np.ndarray, width: int = 40) -> str:
    if pcm_f32 is None or len(pcm_f32) == 0:
        return c(DIM, "·" * width)
    step  = max(1, len(pcm_f32) // width)
    cols  = [float(np.sqrt(np.mean(pcm_f32[i:i+step]**2)))
             for i in range(0, len(pcm_f32), step)][:width]
    mx    = max(cols) if cols else 0.0
    mx    = mx or 1e-9    # guard against all-zero silent chunks
    bars  = "▁▂▃▄▅▆▇█"
    out   = ""
    for v in cols:
        idx = min(int((v / mx) * (len(bars) - 1)), len(bars) - 1)
        out += c(MAG, bars[idx])
    return out + " " * (width - len(cols))

# ── Latency bars ──────────────────────────────────────────────────────────────

def _bar(val, mx, w=24):
    if not mx or val is None: return c(DIM, "░" * w)
    f   = min(int((val / mx) * w), w)
    col = GRN if val < 200 else YLW if val < 500 else MAG if val < 1000 else RED
    return c(col, "█" * f) + c(DIM, "░" * (w - f))

def _ms(ms):
    if ms is None: return c(DIM, "    —   ")
    col = GRN+BOLD if ms<200 else YLW+BOLD if ms<500 else MAG+BOLD if ms<1000 else RED+BOLD
    return c(col, f"{ms:>6.0f}ms")

def _icon(ms):
    if ms is None: return "○"
    return c(GRN,"●") if ms<200 else c(YLW,"●") if ms<500 else c(MAG,"●") if ms<1000 else c(RED,"●")

# ── Print helpers ─────────────────────────────────────────────────────────────

def _header():
    print()
    print(c(BGDARK, CYN+BOLD, " " * TW))
    print(c(BGDARK, CYN+BOLD, f"  ◆  VOICE PIPELINE MONITOR  ◆".ljust(TW)))
    print(c(BGDARK, DIM,      f"  gw:{GW_URL}   tts:Azure   mic:{MIC_RATE}Hz  out:{OUT_RATE}Hz".ljust(TW)))
    print(c(BGDARK, CYN+BOLD, " " * TW))
    print()

def _conv_entry(role, text):
    if role == "user":
        tag = c(BGCYN, BLK+BOLD, " USER "); col = CYN
    else:
        tag = c(BGMAG, WHT+BOLD, "  AI  "); col = MAG
    wrapped = ("\n" + " "*10).join(textwrap.wrap(text, TW-12))
    print(f"\n  {tag}  {c(col, wrapped)}")

def _thinking_banner(tid):
    print(f"\n{SEP}")
    print(f"  {c(YLW+BOLD,'THINKING')}  {c(DIM,'['+tid[:8]+']')}")
    print(SEP)

def _segment_banner(text):
    print()
    print(SEP)
    print(f"  {c(GRN+BOLD,'USER SAID')}   {c(GRN, chr(34)+text+chr(34))}")
    print(SEP)

def _tts_chunk_line(idx, text):
    print(f"\n  {c(MAG+BOLD,'['+str(idx)+']')}  {c(MAG, text)}")

def _latency_panel(ev):
    barge = ev.get("barge_in", False)
    stt   = ev.get("stt_latency_ms")
    cag   = ev.get("cag_first_token_ms")
    c2t   = ev.get("cag_to_tts_ms")
    tts_s = ev.get("tts_synth_ms")
    e2e   = ev.get("e2e_ms")
    toks  = ev.get("total_tokens", 0)
    tid   = ev.get("turn_id","")[:8]
    label = c(RED+BOLD,"INTERRUPTED") if barge else c(GRN+BOLD,"COMPLETE")
    mx    = max((v for v in [stt,cag,c2t,tts_s] if v), default=1)
    print(f"\n{SEP_BOLD}")
    print(f"  {c(BOLD,'LATENCY REPORT')}  {label}  {c(DIM,'['+tid+']')}  {c(DIM,str(toks)+' tokens')}")
    print(SEP_BOLD)
    for lbl, ms, desc in [
        ("STT  word → segment  ", stt,  "Last spoken word until STT fires"),
        ("CAG  seg  → token    ", cag,  "STT segment until first AI token"),
        ("CAG  tok  → TTS send ", c2t,  "First token until first TTS chunk queued"),
        ("TTS  send → audio    ", tts_s,"TTS chunk queued until audio arrives"),
    ]:
        print(f"  {c(DIM,lbl)}  {_ms(ms)}  {_bar(ms,mx)}  {c(DIM,desc)}")
    print(f"\n  {c(BOLD,'E2E  word → audio     ')}  {_ms(e2e)}  {_icon(e2e)}")
    chunks = ev.get("tts_chunks", [])
    if chunks:
        mxc = max((ch.get("synthesis_latency_ms") or 0) for ch in chunks) or 1
        print(f"\n  {c(DIM,'Chunk  Synth dur   Synth lat   Bar                 Audio dur')}")
        print(f"  {c(DIM,'─'*(TW-4))}")
        for ch in chunks:
            i    = ch.get("chunk_index", 0)
            clag = ch.get("synthesis_latency_ms") or 0
            cdur = ch.get("synth_duration_ms")    or 0
            dsec = ch.get("duration_sec")          or 0.0
            star = c(CYN+BOLD,"FIRST") if i==0 else c(DIM,f"+{ch.get('first_chunk_latency_ms',0):.0f}ms")
            print(f"  {c(CYN,'['+str(i)+']')}  {_ms(cdur)}    {_ms(clag)}  {_bar(clag,mxc,20)}  {c(DIM,f'{dsec:.2f}s')}  {star}")
    print(SEP_BOLD)

def _session_summary(summary):
    print(f"\n{SEP_BOLD}")
    print(c(BOLD+CYN,"   SESSION SUMMARY"))
    print(SEP_BOLD)
    print(f"  Turns     : {c(BOLD, str(summary.get('turns',0)))}")
    print(f"  Barge-ins : {c(BOLD, str(summary.get('barge_ins',0)))}")
    for key, lbl in [("stt","STT  word→seg"),("cag","CAG  →token"),
                     ("tts_synth","TTS  →audio"),("e2e","E2E  word→audio")]:
        st = summary.get(key,{})
        if not st: continue
        avg = st.get("avg"); p95 = st.get("p95")
        mn  = st.get("min"); mx2 = st.get("max")
        print(f"\n  {c(BOLD,lbl)}\n    avg {_ms(avg)}  p95 {_ms(p95)}  min {_ms(mn)}  max {_ms(mx2)}\n    {_bar(avg,2000,20)}")
    print(SEP_BOLD)

# ── Continuous audio output thread ────────────────────────────────────────────

def _audio_output_thread():
    global _ai_speaking
    BLOCK   = 1024
    silence = np.zeros(BLOCK, dtype=np.float32)

    try:
        with sd.OutputStream(
            samplerate=OUT_RATE, channels=1, dtype="float32",
            blocksize=BLOCK, latency="low",
        ) as stream:
            leftover = np.zeros(0, dtype=np.float32)

            while not _stop.is_set():
                chunks = []
                try:
                    while True:
                        chunks.append(_pcm_play_q.get_nowait())
                except stdlib_queue.Empty:
                    pass

                if chunks:
                    _ai_speaking = True
                    buf = (np.concatenate([leftover] + chunks)
                           if leftover.size else np.concatenate(chunks))
                    leftover = np.zeros(0, dtype=np.float32)
                    pos = 0
                    while pos < len(buf):
                        frame = buf[pos:pos+BLOCK]
                        if len(frame) < BLOCK:
                            leftover = frame
                            break
                        stream.write(frame)
                        pos += BLOCK
                else:
                    _ai_speaking = False
                    stream.write(silence)
                    import time; time.sleep(0.005)

    except Exception as e:
        print(f"\n  {c(RED,'Audio output error: '+str(e))}")

# ── Waveform display coroutine ────────────────────────────────────────────────

async def _waveform_display():
    last_pcm: np.ndarray = np.zeros(512, dtype=np.float32)
    while not _stop.is_set():
        await asyncio.sleep(0.08)
        if _ai_speaking:
            wf = _waveform_line(last_pcm, width=TW - 20)
            print(f"\r{ts()}  {c(MAG+BOLD,'♪')}  {wf}  ", end="", flush=True)

# ── Mic callback ─────────────────────────────────────────────────────────────

def _mic_cb(indata, frames, time_info, status):
    chunk = indata[:, 0].copy().astype(np.float32)
    i16   = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
    frame = b"\x01" + i16.tobytes()
    if _loop and not _loop.is_closed():
        asyncio.run_coroutine_threadsafe(_audio_in_q.put(frame), _loop)

# ── Sender ────────────────────────────────────────────────────────────────────

async def _sender(ws):
    while not _stop.is_set():
        try:
            frame = await asyncio.wait_for(_audio_in_q.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        try:
            await ws.send(frame)
        except Exception:
            break

# ── Receiver ─────────────────────────────────────────────────────────────────

async def _receiver(ws):
    global _cur_user_words, _cur_ai_tokens, _cur_ai_sentences, _ai_speaking

    _cur_user_words   = []
    _cur_ai_tokens    = []
    _cur_ai_sentences = []
    _in_tok           = False

    async for msg in ws:

        # ── PCM audio bytes from gateway (Azure TTS) ─────────────────────
        if isinstance(msg, bytes):
            if msg:
                pcm = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                _pcm_play_q.put(pcm)
                wf = _waveform_line(pcm, width=TW - 20)
                print(f"\r{ts()}  {c(MAG+BOLD,'♪')}  {wf}  ", end="", flush=True)
            continue

        # ── JSON control frames ───────────────────────────────────────────────
        try:
            ev = json.loads(msg)
        except Exception:
            continue

        t = ev.get("type")

        if DEBUG_RAW:
            print(f"\n{ts()}  {c(DIM,'RAW['+str(t)+']')}  {c(DIM,json.dumps(ev)[:160])}")

        # STT
        if t == "word":
            if _ai_speaking: print()
            _cur_user_words.append(ev.get("word",""))
            print(f"\r{ts()}  🎤  {c(GRN,' '.join(_cur_user_words))}  ", end="", flush=True)

        elif t == "partial":
            word = ev.get("word","").strip().rstrip("?.!,;:")
            if word and (not _cur_user_words or _cur_user_words[-1].lower()!=word.lower()):
                _cur_user_words.append(word)
            print(f"\r{ts()}  🎤  {c(GRN,' '.join(_cur_user_words))}  ", end="", flush=True)

        elif t == "segment":
            text = ev.get("text","").strip()
            print()
            _segment_banner(text)
            _cur_user_words = []

        # CAG
        elif t == "thinking":
            if _in_tok: print(); _in_tok = False
            _cur_ai_tokens = []; _cur_ai_sentences = []
            _thinking_banner(ev.get("turn_id",""))
            print(f"{ts()}  {c(BLU+BOLD,'AI')}  ", end="", flush=True)
            _in_tok = True

        elif t == "ai_token":
            print(c(BLU, ev.get("token","")), end="", flush=True)
            _cur_ai_tokens.append(ev.get("token",""))
            _in_tok = True

        elif t == "ai_sentence":
            text = ev.get("text","").strip()
            _cur_ai_sentences.append(text)
            if _in_tok: print(); _in_tok = False
            _tts_chunk_line(len(_cur_ai_sentences), text)
            print(f"{ts()}  {c(BLU+BOLD,'AI')}  ", end="", flush=True)
            _in_tok = True
            # TTS is handled server-side by the gateway (Azure TTS)

        elif t == "done":
            if _in_tok: print(); _in_tok = False
            full = "".join(_cur_ai_tokens).strip()
            if full:
                _conv_entry("ai", full)
            _cur_ai_tokens = []; _cur_ai_sentences = []

        # Latency
        elif t == "latency":
            if _in_tok: print(); _in_tok = False
            _latency_panel(ev)

        elif t == "session_summary":
            _session_summary(ev.get("latency",{}))
            _stop.set()

        # Misc
        elif t == "barge_in":
            if _in_tok: print(); _in_tok = False
            print(f"\n  {c(YLW+BOLD,'⚡ BARGE-IN')}  {c(YLW,'Stopping AI speech')}")

        elif t == "hallucination_reset":
            if _in_tok: print(); _in_tok = False
            print(f"\n  {c(YLW+BOLD,'HALLUCINATION GUARD')}  {c(YLW,ev.get('detail',''))}")
            _cur_user_words = []

        elif t == "session_reset":
            print(f"\n  {c(YLW,'Session reset: '+ev.get('reason',''))}")

        elif t == "error":
            if _in_tok: print(); _in_tok = False
            print(f"\n  {c(RED+BOLD,'ERROR')}  {c(RED, ev.get('detail',''))}")

        elif t == "session":
            global _session_id
            _session_id = ev.get("session_id", _session_id)
            print(f"\n  {c(CYN+BOLD,'SESSION')}  {c(CYN, _session_id or '?')}")

        elif t == "history":
            role = ev.get("role", "?")
            content = ev.get("content", "")
            _conv_entry(role, content)

        elif t in ("ping", "ready", "ai_state"):
            if t == "ready":
                print(f"\n  {c(GRN+BOLD, '✓ Pipeline ready — start speaking')}\n")

# ── Auth / Session helpers ────────────────────────────────────────────────────

async def _do_auth() -> tuple[Optional[str], Optional[str]]:
    """Register (if --register) + Login → return (token, user_id) or (None, None)."""
    if not args.email or not args.password:
        return None, None

    async with httpx.AsyncClient(timeout=10) as client:
        # Register
        if args.register:
            username = args.username or args.email.split("@")[0]
            print(f"  {c(DIM,'Registering')}  {c(BOLD, args.email)}  …", end=" ", flush=True)
            r = await client.post(f"{AUTH_URL}/auth/register", json={
                "email": args.email, "username": username, "password": args.password,
            })
            if r.status_code == 201:
                print(c(GRN+BOLD, "✓"))
            else:
                print(c(RED, f"FAILED ({r.status_code}: {r.text[:80]})"))
                return None, None

        # Login
        print(f"  {c(DIM,'Logging in ')}  {c(BOLD, args.email)}  …", end=" ", flush=True)
        r = await client.post(f"{AUTH_URL}/auth/login", json={
            "email": args.email, "password": args.password,
        })
        if r.status_code == 200:
            data = r.json()
            token = data["access_token"]
            # Decode user_id from token verify
            r2 = await client.post(f"{AUTH_URL}/auth/verify-token",
                                   headers={"Authorization": f"Bearer {token}"})
            user_id = r2.json().get("id") if r2.status_code == 200 else None
            print(c(GRN+BOLD, f"✓  token={token[:20]}…"))
            return token, user_id
        else:
            print(c(RED, f"FAILED ({r.status_code}: {r.text[:80]})"))
            return None, None


async def _create_or_resume_session(token: str, user_id: str) -> Optional[str]:
    """Create a new session or resume an existing one."""
    async with httpx.AsyncClient(timeout=10) as client:
        if args.session:
            # Verify it exists
            r = await client.get(f"{SESS_URL}/sessions/{args.session}")
            if r.status_code == 200:
                data = r.json()
                print(f"  {c(DIM,'Resuming  ')}  session {c(BOLD, args.session[:12]+'…')}  title={c(CYN, data.get('title','(none)'))}")
                return args.session
            else:
                print(f"  {c(YLW,'Session not found — creating new one')}")

        # Create new
        r = await client.post(f"{SESS_URL}/sessions", json={"user_id": user_id})
        if r.status_code == 201:
            sid = r.json()["id"]
            print(f"  {c(DIM,'Session   ')}  {c(BOLD, sid)}  {c(GRN,'(new)')}")
            return sid
        else:
            print(f"  {c(RED, f'Session creation failed: {r.status_code}')}")
            return None


async def _fetch_and_show_history():
    """Fetch conversation history from messages service and display it."""
    if not _session_id:
        return

    print(f"\n{SEP_BOLD}")
    print(c(BOLD+CYN, "   CONVERSATION HISTORY"))
    print(SEP_BOLD)
    print(f"  {c(DIM,'Session')}  {c(BOLD, _session_id)}")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Fetch session details
            r = await client.get(f"{SESS_URL}/sessions/{_session_id}")
            if r.status_code == 200:
                sess = r.json()
                title = sess.get("title", "(untitled)")
                print(f"  {c(DIM,'Title  ')}  {c(BOLD, title)}")
                print(f"  {c(DIM,'Updated')}  {c(DIM, sess.get('updated_at','?'))}")

            # Fetch messages
            r = await client.get(f"{MSG_URL}/sessions/{_session_id}/messages",
                                 params={"limit": 100})
            if r.status_code == 200:
                messages = r.json()
                if messages:
                    print(f"  {c(DIM,'Messages')} {c(BOLD, str(len(messages)))}")
                    print(SEP)
                    for m in messages:
                        role    = m.get("role", "?")
                        content = m.get("content", "")
                        ts_str  = m.get("created_at", "")[:19]

                        if role == "user":
                            tag = c(BGCYN, BLK+BOLD, " USER  ")
                            col = CYN
                        else:
                            tag = c(BGMAG, WHT+BOLD, " AGENT ")
                            col = MAG

                        wrapped = ("\n" + " "*12).join(textwrap.wrap(content, TW-14))
                        print(f"\n  {tag}  {c(col, wrapped)}")
                        print(f"           {c(DIM, ts_str)}")
                    print(f"\n{SEP}")
                else:
                    print(f"\n  {c(YLW, 'No messages saved yet (CAG might be offline)')}")
            else:
                print(f"  {c(RED, f'Failed to fetch messages: {r.status_code}')}")

            # Show all sessions for user
            if _user_id:
                r = await client.get(f"{SESS_URL}/users/{_user_id}/sessions")
                if r.status_code == 200:
                    sessions = r.json()
                    if len(sessions) > 1:
                        print(f"\n  {c(DIM, f'All sessions for this user ({len(sessions)}):' )}")
                        for s in sessions:
                            marker = c(GRN+BOLD, '→ ') if s['id'] == _session_id else '  '
                            print(f"    {marker}{c(CYN, s['id'][:12]+'…')}  {c(DIM, s.get('title','(untitled)'))}  {c(DIM, s.get('updated_at','')[:19])}")
    except Exception as e:
        print(f"  {c(RED, f'Error fetching history: {e}')}")

    print(SEP_BOLD)


# ── Main ──────────────────────────────────────────────────────────────────────

async def _main():
    global _audio_in_q, _pcm_play_q, _loop, _jwt_token, _user_id, _session_id

    _audio_in_q = asyncio.Queue(maxsize=500)
    _loop       = asyncio.get_running_loop()

    _header()
    print(f"  {c(DIM,'Gateway')}  {c(BOLD,GW_URL)}")
    print(f"  {c(DIM,'TTS    ')}  {c(BOLD,'Azure (server-side)')}")
    print()

    # ── Step 1: Auth ──────────────────────────────────────────────────────
    if args.email and args.password:
        print(SEP)
        print(f"  {c(BOLD+CYN, 'AUTH')}")
        print(SEP)
        _jwt_token, _user_id = await _do_auth()
        if _jwt_token and _user_id:
            # ── Step 2: Session ────────────────────────────────────────────
            _session_id = await _create_or_resume_session(_jwt_token, _user_id)
        print()
    else:
        print(f"  {c(YLW, 'No --email/--password → anonymous mode (no message persistence)')}")
        print()

    # ── Build WS URL with auth params ─────────────────────────────────────
    ws_url = GW_URL
    params = []
    if _jwt_token:
        params.append(f"token={_jwt_token}")
    if _session_id:
        params.append(f"session_id={_session_id}")
    if params:
        ws_url += "?" + "&".join(params)

    print(f"  {c(YLW,'Speak naturally. Barge in any time to interrupt the AI.')}")
    print(f"  {c(DIM,'Ctrl+C to quit — conversation history will be shown.')}\n")
    print(SEP)

    out_thread = threading.Thread(target=_audio_output_thread, daemon=True)
    out_thread.start()

    try:
        async with websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=30,
            max_size=10 * 1024 * 1024,
        ) as ws:
            print(f"\n  {c(GRN+BOLD,'Connected')}", end="")
            if _session_id:
                print(f"  {c(DIM,'session='+ _session_id[:12]+'…')}", end="")
            print(f"\n{SEP_BOLD}")

            with sd.InputStream(
                samplerate=MIC_RATE, channels=1, dtype="float32",
                blocksize=CHUNK_SAMP, callback=_mic_cb,
            ):
                print(f"\n  {c(GRN+BOLD,'Mic open — start speaking...')}\n")
                await asyncio.gather(
                    _sender(ws),
                    _receiver(ws),
                )

    except ConnectionRefusedError:
        print(c(RED, f"\n  Cannot connect to {GW_URL}"))
        print(c(DIM, "     Start the gateway first: python gateway.py"))
        sys.exit(1)
    except websockets.exceptions.ConnectionClosedOK:
        print(c(YLW, "\n  Gateway closed the connection."))
    except Exception as e:
        print(c(RED, f"\n  {e}"))
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        # Always show history on exit
        await _fetch_and_show_history()


if __name__ == "__main__":
    _pcm_play_q  = stdlib_queue.Queue(maxsize=2000)
    _ai_speaking = False

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        _stop.set()
        # Fetch history before exiting
        try:
            asyncio.run(_fetch_and_show_history())
        except Exception:
            pass
        print(c(YLW, "\n\n  Stopped.\n"))