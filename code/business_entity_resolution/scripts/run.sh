#!/usr/bin/env bash
# Run a pipeline python script under a memory watchdog.
#   scripts/run.sh <log-name> <python args...>
# Environment (all optional): ER_PYLIB (extra site dir), ER_MEM_GB (kill threshold, default 19)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_NAME="$1"; shift
LOG_DIR="${ER_LOG_DIR:-/root/er/logs}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="${ER_PYLIB:-/root/pylib}:$ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export ARROW_DEFAULT_MEMORY_POOL=system
LIMIT_GB="${ER_MEM_GB:-17}"

setsid nohup python3 "$@" > "$LOG_DIR/$LOG_NAME.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG_DIR/$LOG_NAME.pid"
# watchdog: total RSS of the process tree must stay below LIMIT_GB
setsid nohup bash -c '
  PID=$1; LIMIT_GB=$2; LOG=$3
  while kill -0 "$PID" 2>/dev/null; do
    TOTAL=$(ps -o rss= --ppid "$PID" -p "$PID" 2>/dev/null | awk "{s+=\$1} END {print s+0}")
    for C in $(pgrep -P "$PID" 2>/dev/null); do
      TOTAL=$((TOTAL + $(ps -o rss= --ppid "$C" 2>/dev/null | awk "{s+=\$1} END {print s+0}")))
    done
    GB=$((TOTAL / 1024 / 1024))
    if [ "$GB" -ge "$LIMIT_GB" ]; then
      echo "$(date +%T) WATCHDOG: RSS ${GB}GB >= ${LIMIT_GB}GB, killing $PID" >> "$LOG"
      pkill -9 -P "$PID"; kill -9 "$PID"
      break
    fi
    sleep 2
  done
' _ "$PID" "$LIMIT_GB" "$LOG_DIR/$LOG_NAME.log" > /dev/null 2>&1 < /dev/null &
echo "started pid $PID -> $LOG_DIR/$LOG_NAME.log"
