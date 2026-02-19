#!/usr/bin/env bash
# llama_server_ctl.sh
#
# Starts a llama.cpp "llama-server" in the background, waits until it is ready,
# and prints the server PID (so you can stop it later).
#
# Works locally and in GitHub Actions (also writes outputs to $GITHUB_OUTPUT if set).

set -euo pipefail

cmd="${1:-start}"

# ---- Config (override via env vars) ----
MODELS_DIR="${MODELS_DIR:-models}"
MODEL_FILE="${MODEL_FILE:-snowflake-arctic-embed-l-v2.0.F16.gguf}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/$MODEL_FILE}"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"

CTX_SIZE="${CTX_SIZE:-8192}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-1024}"

LOG_FILE="${LOG_FILE:-embeddings.log}"
PIDFILE="${PIDFILE:-.llama-server.${PORT}.pid}"

READY_TIMEOUT_S="${READY_TIMEOUT_S:-300}"
READY_INTERVAL_S="${READY_INTERVAL_S:-1}"

# Extra args you might want to pass through (e.g. "--threads 8")
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ---- Helpers ----
log() { printf '%s\n' "$*" >&2; }

die() { log "ERROR: $*"; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

http_status() {
  # Prints HTTP status code (e.g. 200/503) or 000 on failure.
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -sS -o /dev/null -w "%{http_code}" --max-time 2 "$url" || echo "000"
  elif command -v wget >/dev/null 2>&1; then
    # wget spider prints headers to stderr; parse first HTTP status line.
    wget --server-response --spider -T 2 "$url" 2>&1 \
      | awk '/^  HTTP\// {print $2; exit} END{if (NR==0) print "000"}'
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$url" <<'PY'
import sys, urllib.request
url = sys.argv[1]
try:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=2) as r:
        print(r.status)
except Exception:
    print("000")
PY
  else
    echo "000"
  fi
}

port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then
    # ss output includes header; if any listener rows exist => in use
    ss -ltn "sport = :$port" 2>/dev/null | awk 'NR>1 {found=1} END{exit found?0:1}'
  elif command -v netstat >/dev/null 2>&1; then
    netstat -an 2>/dev/null | grep -E 'LISTEN|LISTENING' | grep -E "[:\.]$port[[:space:]]" >/dev/null 2>&1
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", port))
    s.close()
    sys.exit(1)  # not in use
except OSError:
    sys.exit(0)  # in use
PY
  else
    # Best-effort: if we can't check, assume it's free.
    return 1
  fi
}

write_github_outputs() {
  # If running in GitHub Actions, expose values as step outputs.
  # (The step must have an `id:` to access them.)
  local pid="$1"
  local url="$2"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "llama_pid=$pid"
      echo "llama_url=$url"
      echo "llama_log=$LOG_FILE"
      echo "llama_pidfile=$PIDFILE"
    } >> "$GITHUB_OUTPUT"
  fi
}

start_server() {
  [[ -f "$MODEL_PATH" ]] || die "Model file not found: $MODEL_PATH"
  require_cmd "$LLAMA_SERVER_BIN"

  # If we have a PIDFILE and the process is alive, reuse it (and ensure it's ready).
  if [[ -f "$PIDFILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      local url="http://${HOST}:${PORT}"
      local code
      code="$(http_status "${url}/health")"
      if [[ "$code" == "200" ]]; then
        log "llama-server already running (pid=$existing_pid) at $url"
        write_github_outputs "$existing_pid" "$url"
        printf '%s\n' "$existing_pid"
        return 0
      fi
      log "Found running pid=$existing_pid but health=$code; continuing to start a fresh server..."
    else
      rm -f "$PIDFILE"
    fi
  fi

  # Port availability check
  if port_in_use "$PORT"; then
    die "Port $PORT is already in use. Choose another PORT or stop the process using it."
  fi

  local url="http://${HOST}:${PORT}"
  log "Starting llama-server on $url"
  log "Model: $MODEL_PATH"
  log "Log:   $LOG_FILE"
  : > "$LOG_FILE" || true

  # If we fail while waiting, clean up the background server.
  local pid=""
  cleanup_on_fail() {
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      log "Cleaning up (killing pid=$pid) due to startup failure..."
      kill "$pid" 2>/dev/null || true
    fi
  }
  trap cleanup_on_fail EXIT

  # Start in background. (nohup helps it survive step boundaries in CI shells.)
  nohup "$LLAMA_SERVER_BIN" \
    -m "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --embeddings \
    --ctx-size "$CTX_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    $EXTRA_ARGS \
    >>"$LOG_FILE" 2>&1 &

  pid="$!"
  echo "$pid" > "$PIDFILE"
  log "Spawned llama-server pid=$pid (pidfile: $PIDFILE)"

  # Wait until /health returns 200 (503 while loading) per llama.cpp server docs. :contentReference[oaicite:0]{index=0}
  local deadline=$(( "$(date +%s)" + READY_TIMEOUT_S ))
  while true; do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "llama-server exited before becoming ready. Last log lines:"
      tail -n 80 "$LOG_FILE" >&2 || true
      exit 1
    fi

    local code
    code="$(http_status "${url}/health")"

    if [[ "$code" == "200" ]]; then
      log "Server is ready: $url (pid=$pid)"
      write_github_outputs "$pid" "$url"

      # Success: don't kill on EXIT anymore
      trap - EXIT

      # Print PID ONLY to stdout for easy capture (local + CI).
      printf '%s\n' "$pid"
      return 0
    fi

    if (( "$(date +%s)" >= deadline )); then
      log "Timed out after ${READY_TIMEOUT_S}s waiting for readiness on ${url}/health (last code=$code)."
      log "Last log lines:"
      tail -n 80 "$LOG_FILE" >&2 || true
      exit 1
    fi

    sleep "$READY_INTERVAL_S"
  done
}

stop_server() {
  local pid=""
  if [[ -f "$PIDFILE" ]]; then
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  fi

  if [[ -z "$pid" ]]; then
    die "No PID found (pidfile missing/empty: $PIDFILE). Set PIDFILE or remove it and start again."
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    log "No running process with pid=$pid. Removing stale pidfile."
    rm -f "$PIDFILE"
    return 0
  fi

  log "Stopping llama-server pid=$pid"
  kill "$pid" 2>/dev/null || true

  # Wait a bit for clean shutdown
  for _ in {1..50}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PIDFILE"
      log "Stopped."
      return 0
    fi
    sleep 0.1
  done

  log "Still running; sending SIGKILL..."
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PIDFILE"
  log "Stopped (SIGKILL)."
}

case "$cmd" in
  start) start_server ;;
  stop)  stop_server ;;
  *)
    cat >&2 <<USAGE
Usage:
  $0 start   # starts server, waits until ready, prints PID to stdout
  $0 stop    # stops server using PIDFILE

Key env vars:
  MODELS_DIR, MODEL_FILE, MODEL_PATH, LLAMA_SERVER_BIN, HOST, PORT,
  CTX_SIZE, BATCH_SIZE, UBATCH_SIZE, LOG_FILE, PIDFILE,
  READY_TIMEOUT_S, READY_INTERVAL_S, EXTRA_ARGS
USAGE
    exit 2
    ;;
esac
