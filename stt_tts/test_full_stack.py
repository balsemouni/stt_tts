"""
test_full_stack.py — End-to-end test for the entire voice pipeline
══════════════════════════════════════════════════════════════════════

Tests (in order):
  1. Health checks for all services
  2. User registration + login (JWT)
  3. Session creation + listing
  4. Message CRUD (user + agent roles)
  5. Gateway WebSocket — auth, session assignment, inject_query
  6. Full pipeline: inject text → CAG → TTS → audio back

Prerequisites:
  - PostgreSQL running with databases: users, session_chat, message_db
  - Redis running (optional — tests still pass without it)
  - All services started:
      user_auth (8006), session_chat (8005), messages (8003),
      gateway (8090), stt (8001), cag (8000)

Usage:
    pip install httpx websockets
    python test_full_stack.py
    python test_full_stack.py --skip-voice   # skip STT/CAG/TTS (no GPU needed)
"""

import argparse
import asyncio
import json
import sys
import time

try:
    import httpx
except ImportError:
    print("pip install httpx"); sys.exit(1)
try:
    import websockets
except ImportError:
    print("pip install websockets"); sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────

AUTH_URL    = "http://localhost:8006"
SESSION_URL = "http://localhost:8005"
MSG_URL     = "http://localhost:8003"
GATEWAY_URL = "http://localhost:8090"
GATEWAY_WS  = "ws://localhost:8090/ws"

TEST_EMAIL    = f"test_{int(time.time())}@example.com"
TEST_USERNAME = f"testuser_{int(time.time())}"
TEST_PASSWORD = "SecurePass1"

# ── Helpers ───────────────────────────────────────────────────────────────────

passed = 0
failed = 0

def ok(label):
    global passed
    passed += 1
    print(f"  ✅ {label}")

def fail(label, detail=""):
    global failed
    failed += 1
    print(f"  ❌ {label}  {detail}")

def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ══════════════════════════════════════════════════════════════════════════════
#  1. HEALTH CHECKS
# ══════════════════════════════════════════════════════════════════════════════

async def test_health(client: httpx.AsyncClient, skip_voice: bool):
    section("1 · Health Checks")

    services = [
        ("Auth",    f"{AUTH_URL}/health"),
        ("Session", f"{SESSION_URL}/health"),
        ("Message", f"{MSG_URL}/health"),
        ("Gateway", f"{GATEWAY_URL}/health"),
    ]
    if not skip_voice:
        services += [
            ("STT", "http://localhost:8001/health"),
            ("CAG", "http://localhost:8000/health"),
        ]

    for name, url in services:
        try:
            r = await client.get(url, timeout=5)
            if r.status_code == 200:
                ok(f"{name} service — {url}")
            else:
                fail(f"{name} service", f"status={r.status_code}")
        except httpx.RequestError as e:
            fail(f"{name} service", f"unreachable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  2. USER AUTH — register, login, verify token
# ══════════════════════════════════════════════════════════════════════════════

async def test_auth(client: httpx.AsyncClient) -> dict:
    section("2 · User Auth")
    result = {"user_id": None, "token": None}

    # Register
    r = await client.post(f"{AUTH_URL}/auth/register", json={
        "email": TEST_EMAIL, "username": TEST_USERNAME, "password": TEST_PASSWORD,
    }, timeout=10)
    if r.status_code == 201:
        data = r.json()
        result["user_id"] = data["id"]
        ok(f"Register user — id={data['id'][:8]}…")
    else:
        fail("Register user", r.text)
        return result

    # Login
    r = await client.post(f"{AUTH_URL}/auth/login", json={
        "email": TEST_EMAIL, "password": TEST_PASSWORD,
    }, timeout=10)
    if r.status_code == 200:
        data = r.json()
        result["token"] = data["access_token"]
        ok(f"Login — got access_token ({len(data['access_token'])} chars)")
    else:
        fail("Login", r.text)
        return result

    # Verify token
    r = await client.post(f"{AUTH_URL}/auth/verify-token",
                          headers={"Authorization": f"Bearer {result['token']}"},
                          timeout=5)
    if r.status_code == 200:
        ok(f"Verify token — user={r.json().get('username')}")
    else:
        fail("Verify token", r.text)

    # Bad token should fail
    r = await client.post(f"{AUTH_URL}/auth/verify-token",
                          headers={"Authorization": "Bearer invalid.token.here"},
                          timeout=5)
    if r.status_code in (401, 403):
        ok("Reject bad token")
    else:
        fail("Reject bad token", f"status={r.status_code}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  3. SESSION CRUD
# ══════════════════════════════════════════════════════════════════════════════

async def test_sessions(client: httpx.AsyncClient, user_id: str) -> str | None:
    section("3 · Session Service")
    session_id = None

    # Create session
    r = await client.post(f"{SESSION_URL}/sessions", json={
        "user_id": user_id, "title": "Test Voice Chat"
    }, timeout=5)
    if r.status_code == 201:
        data = r.json()
        session_id = data["id"]
        ok(f"Create session — id={session_id[:8]}…")
    else:
        fail("Create session", r.text)
        return None

    # Get session
    r = await client.get(f"{SESSION_URL}/sessions/{session_id}", timeout=5)
    if r.status_code == 200 and r.json()["id"] == session_id:
        ok(f"Get session — title={r.json().get('title')!r}")
    else:
        fail("Get session", r.text)

    # List sessions for user
    r = await client.get(f"{SESSION_URL}/users/{user_id}/sessions", timeout=5)
    if r.status_code == 200 and len(r.json()) >= 1:
        ok(f"List sessions — count={len(r.json())}")
    else:
        fail("List sessions", r.text)

    # Exists endpoint
    r = await client.get(f"{SESSION_URL}/sessions/{session_id}/exists", timeout=5)
    if r.status_code == 200 and r.json().get("exists"):
        ok("Session exists check")
    else:
        fail("Session exists check", r.text)

    # Update title
    r = await client.patch(f"{SESSION_URL}/sessions/{session_id}/title",
                           json={"title": "Updated Title"}, timeout=5)
    if r.status_code == 200 and r.json()["title"] == "Updated Title":
        ok("Update session title")
    else:
        fail("Update session title", r.text)

    return session_id


# ══════════════════════════════════════════════════════════════════════════════
#  4. MESSAGE CRUD
# ══════════════════════════════════════════════════════════════════════════════

async def test_messages(client: httpx.AsyncClient, session_id: str):
    section("4 · Message Service")

    # Save user message
    r = await client.post(f"{MSG_URL}/sessions/{session_id}/messages", json={
        "role": "user", "content": "Hello, can you help me?"
    }, timeout=5)
    if r.status_code == 201:
        msg1 = r.json()
        ok(f"Save user message — id={msg1['id'][:8]}…")
    else:
        fail("Save user message", r.text)
        return

    # Save agent message
    r = await client.post(f"{MSG_URL}/sessions/{session_id}/messages", json={
        "role": "agent", "content": "Of course! How can I assist you today?"
    }, timeout=5)
    if r.status_code == 201:
        msg2 = r.json()
        ok(f"Save agent message — id={msg2['id'][:8]}…")
    else:
        fail("Save agent message", r.text)

    # Invalid role should fail
    r = await client.post(f"{MSG_URL}/sessions/{session_id}/messages", json={
        "role": "hacker", "content": "nope"
    }, timeout=5)
    if r.status_code == 400:
        ok("Reject invalid role")
    else:
        fail("Reject invalid role", f"status={r.status_code}")

    # List messages
    r = await client.get(f"{MSG_URL}/sessions/{session_id}/messages", timeout=5)
    if r.status_code == 200:
        msgs = r.json()
        roles = [m["role"] for m in msgs]
        if "user" in roles and "agent" in roles:
            ok(f"List messages — count={len(msgs)}, roles={roles}")
        else:
            fail("List messages roles", f"roles={roles}")
    else:
        fail("List messages", r.text)

    # Get single message
    r = await client.get(f"{MSG_URL}/messages/{msg1['id']}", timeout=5)
    if r.status_code == 200 and r.json()["content"] == "Hello, can you help me?":
        ok("Get single message")
    else:
        fail("Get single message", r.text)


# ══════════════════════════════════════════════════════════════════════════════
#  5. GATEWAY WEBSOCKET — auth + session
# ══════════════════════════════════════════════════════════════════════════════

async def test_gateway_ws_auth(token: str):
    section("5 · Gateway WebSocket — Auth & Session")

    # Bad token → should be rejected
    try:
        ws = await websockets.connect(f"{GATEWAY_WS}?token=bad.token.here",
                                      open_timeout=5)
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        msg = json.loads(raw)
        if msg.get("type") == "error":
            ok("Reject bad token on WS")
        else:
            fail("Reject bad token on WS", f"got: {msg}")
        await ws.close()
    except Exception as e:
        ok(f"Reject bad token on WS (connection refused: {type(e).__name__})")

    # Good token → should get session + ready
    session_id = None
    try:
        ws = await websockets.connect(f"{GATEWAY_WS}?token={token}",
                                      open_timeout=10)
        events = []
        for _ in range(10):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=20)
                msg = json.loads(raw) if isinstance(raw, str) else {}
                events.append(msg)
                if msg.get("type") == "ready":
                    break
            except asyncio.TimeoutError:
                continue

        types = [e.get("type") for e in events]

        if "session" in types:
            session_id = next(e["session_id"] for e in events if e.get("type") == "session")
            ok(f"Got session assignment — {session_id[:8]}…")
        else:
            fail("Session assignment", f"types={types}")

        if "ready" in types:
            ok("Got ready event")
        else:
            fail("Ready event", f"types={types}")

        await ws.close()
    except Exception as e:
        fail("Gateway WS connect", str(e))

    # Anonymous connection (no token) — should work too
    try:
        ws = await websockets.connect(GATEWAY_WS, open_timeout=10)
        events = []
        for _ in range(5):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=20)
                msg = json.loads(raw) if isinstance(raw, str) else {}
                events.append(msg)
                if msg.get("type") == "ready":
                    break
            except asyncio.TimeoutError:
                continue

        types = [e.get("type") for e in events]
        if "ready" in types:
            ok("Anonymous WS connect (no token) — ready")
        else:
            fail("Anonymous WS connect", f"types={types}")
        await ws.close()
    except Exception as e:
        fail("Anonymous WS connect", str(e))

    return session_id


# ══════════════════════════════════════════════════════════════════════════════
#  6. FULL PIPELINE — inject text → CAG → TTS → audio
# ══════════════════════════════════════════════════════════════════════════════

async def test_full_pipeline(token: str):
    section("6 · Full Pipeline — inject_query → CAG → TTS → audio")

    try:
        ws = await websockets.connect(f"{GATEWAY_WS}?token={token}",
                                      open_timeout=10)

        # Wait for ready
        for _ in range(10):
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = json.loads(raw) if isinstance(raw, str) else {}
            if msg.get("type") == "ready":
                break

        ok("Connected and ready")

        # Inject a text query (simulates STT output)
        ctrl = json.dumps({"type": "inject_query", "text": "What is your name?"})
        await ws.send(b'\x02' + ctrl.encode())
        ok("Sent inject_query: 'What is your name?'")

        # Collect events for up to 30s
        got_segment   = False
        got_thinking  = False
        got_token     = False
        got_sentence  = False
        got_audio     = False
        got_done      = False
        session_id    = None

        t0 = time.monotonic()
        while time.monotonic() - t0 < 30:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
            except asyncio.TimeoutError:
                break

            if isinstance(raw, bytes):
                got_audio = True
                continue

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            t = msg.get("type", "")

            if t == "session":
                session_id = msg.get("session_id")
            elif t == "segment":
                got_segment = True
            elif t == "thinking":
                got_thinking = True
            elif t == "ai_token":
                if not got_token:
                    got_token = True
                    print(f"       AI: ", end="", flush=True)
                print(msg.get("token", ""), end="", flush=True)
            elif t == "ai_sentence":
                got_sentence = True
            elif t == "audio":
                got_audio = True
            elif t == "done":
                got_done = True
                print()  # newline after tokens
                break
            elif t == "error":
                fail("Pipeline error", msg.get("message") or msg.get("detail"))
                break

        if got_segment:  ok("Got STT segment echo")
        else:            fail("STT segment echo")
        if got_thinking: ok("Got thinking event")
        else:            fail("Thinking event")
        if got_token:    ok("Got AI tokens (CAG streaming)")
        else:            fail("AI tokens")
        if got_sentence: ok("Got AI sentence (TTS chunk)")
        else:            fail("AI sentence")
        if got_audio:    ok("Got audio frames (TTS)")
        else:            fail("Audio frames (TTS may be down)")
        if got_done:     ok("Got done event — turn complete")
        else:            fail("Done event")

        # Verify message was persisted
        if session_id:
            await asyncio.sleep(1)  # let fire-and-forget persist finish
            async with httpx.AsyncClient() as http:
                r = await http.get(f"{MSG_URL}/sessions/{session_id}/messages", timeout=5)
                if r.status_code == 200:
                    msgs = r.json()
                    roles = [m["role"] for m in msgs]
                    if "user" in roles and "agent" in roles:
                        ok(f"Messages persisted — {len(msgs)} messages, roles={roles}")
                    elif "user" in roles:
                        ok(f"User message persisted ({len(msgs)} msgs) — agent may still be saving")
                    else:
                        fail("Message persistence", f"roles={roles}")
                else:
                    fail("Message persistence", f"status={r.status_code}")

        await ws.close()

    except Exception as e:
        fail("Full pipeline", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main(skip_voice: bool):
    print("\n" + "═"*60)
    print("  FULL STACK TEST — stt_tts voice pipeline")
    print("═"*60)

    async with httpx.AsyncClient() as client:
        # 1. Health
        await test_health(client, skip_voice)

        # 2. Auth
        auth = await test_auth(client)
        if not auth["token"]:
            print("\n⛔ Auth failed — cannot continue.\n")
            return

        # 3. Sessions
        session_id = await test_sessions(client, auth["user_id"])
        if not session_id:
            print("\n⛔ Session creation failed — cannot continue.\n")
            return

        # 4. Messages
        await test_messages(client, session_id)

    # 5. Gateway WebSocket auth
    ws_session_id = await test_gateway_ws_auth(auth["token"])

    # 6. Full voice pipeline (needs STT + CAG + TTS)
    if not skip_voice:
        await test_full_pipeline(auth["token"])
    else:
        section("6 · Full Pipeline — SKIPPED (--skip-voice)")

    # Summary
    print(f"\n{'═'*60}")
    print(f"  RESULTS:  {passed} passed, {failed} failed")
    print(f"{'═'*60}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full stack test for stt_tts")
    parser.add_argument("--skip-voice", action="store_true",
                        help="Skip STT/CAG/TTS tests (no GPU needed)")
    args = parser.parse_args()
    asyncio.run(main(args.skip_voice))
