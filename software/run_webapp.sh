#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_FILE="$SCRIPT_DIR/app.py"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"
URL="http://${HOST}:${PORT}"

if [[ ! -f "$APP_FILE" ]]; then
  echo "Error: app.py not found at $APP_FILE" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$SCRIPT_DIR/../eeg_env/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/../eeg_env/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: No Python interpreter found." >&2
    exit 1
  fi
fi

open_browser() {
  if grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
    if command -v powershell.exe >/dev/null 2>&1; then
      powershell.exe -NoProfile -Command "Start-Process '$URL'" >/dev/null 2>&1 || true
      return
    fi
    if command -v cmd.exe >/dev/null 2>&1; then
      cmd.exe /C start "$URL" >/dev/null 2>&1 || true
      return
    fi
  fi

  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
    return
  fi

  if command -v sensible-browser >/dev/null 2>&1; then
    sensible-browser "$URL" >/dev/null 2>&1 || true
  fi
}

wait_for_server() {
  local attempts=120
  local i
  for ((i=1; i<=attempts; i++)); do
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS "$URL/" >/dev/null 2>&1; then
        return 0
      fi
    else
      if "$PYTHON_BIN" -c "import sys,urllib.request; urllib.request.urlopen('$URL/', timeout=0.5); sys.exit(0)" >/dev/null 2>&1; then
        return 0
      fi
    fi

    if ! kill -0 "$APP_PID" >/dev/null 2>&1; then
      return 1
    fi

    if [[ "$HOST" == "127.0.0.1" ]] && command -v curl >/dev/null 2>&1; then
      if curl -fsS "http://localhost:${PORT}/" >/dev/null 2>&1; then
        URL="http://localhost:${PORT}"
        return 0
      fi
    elif [[ "$HOST" == "127.0.0.1" ]]; then
      if "$PYTHON_BIN" -c "import sys,urllib.request; urllib.request.urlopen('http://localhost:${PORT}/', timeout=0.5); sys.exit(0)" >/dev/null 2>&1; then
        URL="http://localhost:${PORT}"
        return 0
      fi
    fi

    if [[ "$HOST" == "localhost" ]] && command -v curl >/dev/null 2>&1; then
      if curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
        URL="http://127.0.0.1:${PORT}"
        return 0
      fi
    elif [[ "$HOST" == "localhost" ]]; then
      if "$PYTHON_BIN" -c "import sys,urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/', timeout=0.5); sys.exit(0)" >/dev/null 2>&1; then
        URL="http://127.0.0.1:${PORT}"
        return 0
      fi
    fi

    if [[ "$i" -eq "$attempts" ]]; then
      break
    fi
    sleep 0.25
  done
  return 1
}

echo "Starting EEG web app with: $PYTHON_BIN $APP_FILE"
(
  cd "$SCRIPT_DIR"
  FLASK_RUN_HOST="$HOST" FLASK_RUN_PORT="$PORT" "$PYTHON_BIN" app.py
) &
APP_PID=$!

cleanup() {
  if kill -0 "$APP_PID" >/dev/null 2>&1; then
    kill "$APP_PID" >/dev/null 2>&1 || true
    wait "$APP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if wait_for_server; then
  echo "Server is up at $URL"
  open_browser
else
  if kill -0 "$APP_PID" >/dev/null 2>&1; then
    echo "Warning: server did not respond at $URL in time." >&2
  else
    echo "Error: Flask process exited before server became reachable." >&2
  fi
fi

wait "$APP_PID"
