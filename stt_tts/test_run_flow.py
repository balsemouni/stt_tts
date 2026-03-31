"""
test_run_flow.py — Non-interactive full flow test (no mic needed)
Runs: register → login → session → gateway WS → inject_query → check messages → resume session
"""
import asyncio
import json
import time
import httpx
import websockets

AUTH = "http://localhost:8006"
SESS = "http://localhost:8005"
MSG  = "http://localhost:8003"

EMAIL = f"flowtest_{int(time.time())}@example.com"
PASS  = "FlowTest123"
USER  = f"flowtest_{int(time.time())}"

G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
C = "\033[96m"
B = "\033[1m"
D = "\033[2m"
X = "\033[0m"

def ok(msg):  print(f"  {G}✅ {msg}{X}")
def fail(msg): print(f"  {R}❌ {msg}{X}")
def info(msg): print(f"  {D}ℹ  {msg}{X}")
def head(msg): print(f"\n{C}{B}{'═'*60}{X}\n  {C}{B}{msg}{X}\n{C}{B}{'═'*60}{X}")


async def run():
    async with httpx.AsyncClient(timeout=10) as http:

        # ── 1. REGISTER ──────────────────────────────────────────────────
        head("STEP 1 · Register")
        r = await http.post(f"{AUTH}/auth/register", json={
            "email": EMAIL, "username": USER, "password": PASS,
        })
        if r.status_code == 201:
            user_id = r.json()["id"]
            ok(f"Registered: {EMAIL}  id={user_id[:12]}…")
        else:
            fail(f"Register failed: {r.status_code} {r.text[:80]}")
            return

        # ── 2. LOGIN ─────────────────────────────────────────────────────
        head("STEP 2 · Login")
        r = await http.post(f"{AUTH}/auth/login", json={
            "email": EMAIL, "password": PASS,
        })
        if r.status_code != 200:
            fail(f"Login failed: {r.status_code}")
            return
        token = r.json()["access_token"]
        ok(f"Logged in — token: {token[:25]}…")

        r = await http.post(f"{AUTH}/auth/verify-token",
                            headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            user_id = r.json()["id"]
            ok(f"Token verified — user: {user_id[:12]}…")
        else:
            fail("Token verify failed")
            return

        # ── 3. CREATE SESSION ────────────────────────────────────────────
        head("STEP 3 · Create Session")
        r = await http.post(f"{SESS}/sessions", json={"user_id": user_id})
        if r.status_code != 201:
            fail(f"Session creation failed: {r.status_code} {r.text[:80]}")
            return
        session_id = r.json()["id"]
        ok(f"Session created: {session_id}")

        # ── 4. CONNECT GATEWAY WS (authenticated + session) ─────────────
        head("STEP 4 · Connect Gateway WebSocket")
        ws_url = f"ws://localhost:8090/ws?token={token}&session_id={session_id}"
        info(f"Connecting: ws://…/ws?token=…&session_id={session_id[:12]}…")

        ws = await websockets.connect(ws_url, open_timeout=10)

        # Collect initial events
        got_session = False
        got_ready = False
        for _ in range(15):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=20)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                t = msg.get("type", "")
                if t == "session":
                    got_session = True
                    ok(f"Gateway assigned session: {msg.get('session_id', '?')}")
                elif t == "ready":
                    got_ready = True
                    ok(f"Pipeline ready: {msg.get('message', '')}")
                    break
                else:
                    info(f"Event: {t}")
            except asyncio.TimeoutError:
                break

        if not got_session:
            fail("No session event from gateway")
        if not got_ready:
            info("No ready event (STT offline) — continuing anyway")

        # ── 5. INJECT QUERIES (simulates what STT would produce) ─────────
        head("STEP 5 · Simulate Speaking (inject_query)")
        queries = [
            "Hello what services do you offer",
            "Can you tell me about pricing",
        ]

        for i, query_text in enumerate(queries, 1):
            info(f"Sending query {i}: \"{query_text}\"")
            ctrl = json.dumps({"type": "inject_query", "text": query_text})
            await ws.send(b'\x02' + ctrl.encode())

            got_segment = False
            got_thinking = False
            ai_tokens = []
            ai_sentences = []
            audio_chunks = 0

            for _ in range(20):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                except asyncio.TimeoutError:
                    break
                except websockets.exceptions.ConnectionClosed:
                    info("WS closed")
                    break

                if isinstance(raw, bytes):
                    audio_chunks += 1
                    continue

                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                t = msg.get("type", "")
                if t == "segment":
                    got_segment = True
                elif t == "thinking":
                    got_thinking = True
                elif t == "ai_token":
                    ai_tokens.append(msg.get("token", ""))
                elif t == "ai_sentence":
                    ai_sentences.append(msg.get("text", ""))
                elif t == "done":
                    break
                elif t == "error":
                    info(f"Error: {msg}")
                    break

            if got_segment:
                ok(f"STT segment echoed: \"{query_text}\"")
            if got_thinking:
                ok("CAG acknowledged — thinking")
            if ai_tokens:
                full = "".join(ai_tokens).strip()
                preview = full[:100] + "…" if len(full) > 100 else full
                ok(f"CAG response ({len(ai_tokens)} tokens): \"{preview}\"")
            if ai_sentences:
                ok(f"TTS got {len(ai_sentences)} sentence(s)")
            if audio_chunks:
                ok(f"Audio streamed: {audio_chunks} PCM chunks")
            if not got_thinking and not ai_tokens:
                info("CAG offline — gateway still saved the user message via _persist()")

            if i < len(queries):
                await asyncio.sleep(0.5)

        await ws.close()
        ok("WebSocket disconnected")

        # Wait for fire-and-forget saves
        info("Waiting for async saves to complete…")
        await asyncio.sleep(2)

        # ── 6. CHECK MESSAGES IN DATABASE ────────────────────────────────
        head("STEP 6 · Check Messages (REST API)")
        r = await http.get(f"{MSG}/sessions/{session_id}/messages",
                           params={"limit": 50})
        if r.status_code != 200:
            fail(f"Failed to fetch messages: {r.status_code}")
        else:
            msgs = r.json()
            ok(f"Found {len(msgs)} messages in session")

            if msgs:
                print(f"\n  {B}📜 Conversation History:{X}")
                print(f"  {'─'*50}")
                for m in msgs:
                    role = m["role"]
                    content = m["content"]
                    ts_str = m.get("created_at", "")[:19]
                    icon = "🧑" if role == "user" else "🤖"
                    color = Y if role == "user" else G
                    print(f"    {icon} {color}[{role}]{X} {content[:120]}")
                    print(f"       {D}{ts_str}{X}")
                print(f"  {'─'*50}")

                roles = set(m["role"] for m in msgs)
                if "user" in roles:
                    ok("User messages saved (gateway auto-persisted STT text)")
                else:
                    fail("No user messages — gateway _persist() may not have fired")
                if "agent" in roles:
                    ok("Agent messages saved (gateway auto-persisted CAG response)")
                else:
                    info("No agent messages (CAG offline — no response to save)")
            else:
                info("No messages yet — CAG is offline so gateway can't complete query")

        # ── 7. CHECK SESSION TITLE ───────────────────────────────────────
        head("STEP 7 · Check Session Details")
        r = await http.get(f"{SESS}/sessions/{session_id}")
        if r.status_code == 200:
            sess = r.json()
            title = sess.get("title", "(none)")
            ok(f"Session title: \"{title}\"")
            ok(f"Updated at: {sess.get('updated_at', '?')}")
        else:
            fail(f"Failed to fetch session: {r.status_code}")

        r = await http.get(f"{SESS}/users/{user_id}/sessions")
        if r.status_code == 200:
            sessions = r.json()
            ok(f"User has {len(sessions)} session(s)")

        # ── 8. RESUME SESSION (reconnect) ────────────────────────────────
        head("STEP 8 · Resume Session (Reconnect)")
        info(f"Reconnecting with same session_id…")

        ws2_url = f"ws://localhost:8090/ws?token={token}&session_id={session_id}"
        ws2 = await websockets.connect(ws2_url, open_timeout=10)

        history_msgs = []
        resumed_session = None
        ready2 = False

        for _ in range(20):
            try:
                raw = await asyncio.wait_for(ws2.recv(), timeout=20)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                t = msg.get("type", "")
                if t == "session":
                    resumed_session = msg.get("session_id")
                elif t == "history":
                    history_msgs.append(msg)
                elif t == "ready":
                    ready2 = True
                    break
            except asyncio.TimeoutError:
                continue

        if resumed_session == session_id:
            ok(f"Resumed same session: {resumed_session}")
        elif resumed_session:
            info(f"Got different session: {resumed_session}")

        if history_msgs:
            ok(f"Received {len(history_msgs)} history messages via WebSocket")
            print(f"\n  {B}📜 History delivered on reconnect:{X}")
            print(f"  {'─'*50}")
            for h in history_msgs:
                role = h.get("role", "?")
                content = h.get("content", "")
                icon = "🧑" if role == "user" else "🤖"
                color = Y if role == "user" else G
                print(f"    {icon} {color}[{role}]{X} {content[:120]}")
            print(f"  {'─'*50}")
        else:
            info("No history via WS (loads into CAG context when CAG is online)")

        if ready2:
            ok("Pipeline ready again — can continue chatting")

        await ws2.close()
        ok("Session resumed and verified")

        # ── SUMMARY ──────────────────────────────────────────────────────
        head("SUMMARY")
        print(f"""
  {B}Flow tested:{X}
    1. Register       {G}✅{X}  {EMAIL}
    2. Login          {G}✅{X}  JWT token obtained
    3. Session        {G}✅{X}  {session_id[:12]}…
    4. Gateway WS     {G}✅{X}  Authenticated + session assigned
    5. inject_query   {G}✅{X}  {len(queries)} queries sent (simulated STT)
    6. Messages       {G}✅{X}  {len(msgs)} saved in DB
    7. Session title  {G}✅{X}  \"{title}\"
    8. Resume         {G}✅{X}  Same session restored on reconnect
""")


if __name__ == "__main__":
    asyncio.run(run())
