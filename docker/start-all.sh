#!/bin/bash
# No set -e — STT/CAG may fail without GPU, other services must keep running

echo "═══════════════════════════════════════════════════════════"
echo "  AskNova AI — All-in-One Container Startup"
echo "═══════════════════════════════════════════════════════════"

# ── 1. Start PostgreSQL ──────────────────────────────────────────────
echo "[1/5] Starting PostgreSQL..."
mkdir -p /var/run/postgresql && chown postgres:postgres /var/run/postgresql
touch /tmp/postgresql.log && chown postgres:postgres /tmp/postgresql.log
su postgres -c "/usr/lib/postgresql/16/bin/pg_ctl start -D /var/lib/postgresql/16/main -l /tmp/postgresql.log -w -t 30"

# Wait for PG
for i in $(seq 1 30); do
    su postgres -c "/usr/lib/postgresql/16/bin/pg_isready -q" && break
    sleep 1
done

# Create databases if they don't exist
su postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname = 'users'\" | grep -q 1 || psql -c 'CREATE DATABASE users;'"
su postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname = 'session_chat'\" | grep -q 1 || psql -c 'CREATE DATABASE session_chat;'"
su postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname = 'message_db'\" | grep -q 1 || psql -c 'CREATE DATABASE message_db;'"
echo "[1/5] PostgreSQL ready."

# ── 2. Start Redis ───────────────────────────────────────────────────
echo "[2/5] Starting Redis..."
redis-server --daemonize yes --protected-mode no
sleep 1
redis-cli ping > /dev/null
echo "[2/5] Redis ready."

# ── 3. Start Python Backend Services ────────────────────────────────
echo "[3/5] Starting backend services..."

# Auth (8006)
cd /app/user_auth
DATABASE_USER=postgres DATABASE_PASSWORD=asknova123 DATABASE_HOST=localhost DATABASE_PORT=5432 \
    python main.py &
AUTH_PID=$!

# Sessions (8005)
cd /app/session_chat
DATABASE_USER=postgres DATABASE_PASSWORD=asknova123 DATABASE_HOST=localhost DATABASE_PORT=5432 \
    USER_SERVICE_URL=http://localhost:8006 REDIS_HOST=localhost \
    python main.py &
SESSION_PID=$!

# Messages (8003)
cd /app/messages
DATABASE_USER=postgres DATABASE_PASSWORD=asknova123 DATABASE_HOST=localhost DATABASE_PORT=5432 \
    SESSION_SERVICE_URL=http://localhost:8005 \
    python main.py &
MSG_PID=$!

# Wait for auth/sessions/messages to be up
echo "  Waiting for backend services..."
for i in $(seq 1 30); do
    (curl -sf http://localhost:8006/docs > /dev/null 2>&1) && break
    sleep 1
done

# Gateway (8090) — connects to STT/CAG on localhost
cd /app
STT_WS_URL=ws://localhost:8001/stream/mux \
    CAG_WS_URL=ws://localhost:8000/chat/ws \
    USER_SERVICE_URL=http://localhost:8006 \
    SESSION_SERVICE_URL=http://localhost:8005 \
    MESSAGE_SERVICE_URL=http://localhost:8003 \
    AZURE_TTS_KEY="${AZURE_TTS_KEY:-}" \
    AZURE_TTS_ENDPOINT="${AZURE_TTS_ENDPOINT:-https://francecentral.tts.speech.microsoft.com/cognitiveservices/v1}" \
    AZURE_TTS_VOICE="${AZURE_TTS_VOICE:-en-US-AriaNeural}" \
    python -m uvicorn gateway.gateway:app --host 0.0.0.0 --port 8090 --workers 1 &
GW_PID=$!

echo "[3/5] Backend services started."

# ── 4. Start STT & CAG (GPU services) ───────────────────────────────
echo "[4/5] Starting AI services (STT + CAG)..."

# STT (8001) — requires GPU, will fail gracefully without one
cd /app/stt
STT_HOST=0.0.0.0 STT_PORT=8001 DEVICE=cuda \
    python main.py &
STT_PID=$!

# CAG (8000) — requires GPU, will fail gracefully without one
cd /app/cag
PORT=8000 \
    python main.py &
CAG_PID=$!

echo "[4/5] AI services starting (model loading may take 30-60s)..."

# ── 5. Start Frontend ───────────────────────────────────────────────
echo "[5/5] Starting frontend..."
cd /app/frontend
npx tsx server.ts &
FE_PID=$!

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  AskNova AI — All Services Running"
echo "═══════════════════════════════════════════════════════════"
echo "  Frontend:   http://localhost:3000"
echo "  Auth:       http://localhost:8006"
echo "  Sessions:   http://localhost:8005"
echo "  Messages:   http://localhost:8003"
echo "  Gateway:    ws://localhost:8090"
echo "  STT:        ws://localhost:8001"
echo "  CAG:        ws://localhost:8000"
echo "  PostgreSQL: localhost:5432"
echo "  Redis:      localhost:6379"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Keep container alive forever
# Monitor core services (ignore STT/CAG which need GPU)
while true; do
    sleep 30
    # Auto-restart core services if they die
    kill -0 $AUTH_PID 2>/dev/null || { echo "[watchdog] Restarting auth..."; cd /app/user_auth; DATABASE_USER=postgres DATABASE_PASSWORD=asknova123 DATABASE_HOST=localhost DATABASE_PORT=5432 python main.py & AUTH_PID=$!; }
    kill -0 $SESSION_PID 2>/dev/null || { echo "[watchdog] Restarting sessions..."; cd /app/session_chat; DATABASE_USER=postgres DATABASE_PASSWORD=asknova123 DATABASE_HOST=localhost DATABASE_PORT=5432 USER_SERVICE_URL=http://localhost:8006 REDIS_HOST=localhost python main.py & SESSION_PID=$!; }
    kill -0 $MSG_PID 2>/dev/null || { echo "[watchdog] Restarting messages..."; cd /app/messages; DATABASE_USER=postgres DATABASE_PASSWORD=asknova123 DATABASE_HOST=localhost DATABASE_PORT=5432 SESSION_SERVICE_URL=http://localhost:8005 python main.py & MSG_PID=$!; }
    kill -0 $GW_PID 2>/dev/null || { echo "[watchdog] Restarting gateway..."; cd /app; STT_WS_URL=ws://localhost:8001/stream/mux CAG_WS_URL=ws://localhost:8000/chat/ws USER_SERVICE_URL=http://localhost:8006 SESSION_SERVICE_URL=http://localhost:8005 MESSAGE_SERVICE_URL=http://localhost:8003 python -m uvicorn gateway.gateway:app --host 0.0.0.0 --port 8090 --workers 1 & GW_PID=$!; }
    kill -0 $FE_PID 2>/dev/null || { echo "[watchdog] Restarting frontend..."; cd /app/frontend; npx tsx server.ts & FE_PID=$!; }
done
