#!/usr/bin/env bash
# Launch or resume the bounded experiment independently of an interactive shell.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
RUN_DIR="${1:-runs/small_v6}"
CONFIG_PATH="${2:-configs/small.yaml}"
RESEARCH_COMMAND="${3:-discover}"
case "$RESEARCH_COMMAND" in
    prepare|discover|adapt|evaluate|run|stage2|resume) ;;
    *) echo "Command must be prepare, discover, adapt, evaluate, run, stage2 or resume" >&2; exit 1 ;;
esac
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
PID_FILE="$RUN_DIR/research.pid"
exec 9>"$RUN_DIR/launcher.lock"
flock -n 9 || { echo "Another launcher is active for $RUN_DIR" >&2; exit 1; }
if [[ -f "$PID_FILE" ]]; then
    read -r RESEARCH_PID < "$PID_FILE"
    if [[ "$RESEARCH_PID" =~ ^[0-9]+$ ]] && kill -0 "$RESEARCH_PID" 2>/dev/null; then
        PROCESS_COMMAND="$(tr '\0' ' ' < "/proc/$RESEARCH_PID/cmdline" 2>/dev/null || true)"
        if [[ "$PROCESS_COMMAND" == *simex* && "$PROCESS_COMMAND" == *--run* ]]; then
            echo "Already running: PID $RESEARCH_PID; log: $RUN_DIR/console.log"
            exit 0
        fi
    fi
fi
[[ -x "$PROJECT_DIR/.venv/bin/simex" ]] || { echo "Run scripts/setup.sh first" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "Configuration not found: $CONFIG_PATH" >&2; exit 1; }
if command -v systemd-run >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
    RUN_KEY="$(printf '%s' "$RUN_DIR" | sha256sum)"
    SERVICE_NAME="simex-${RUN_KEY:0:12}.service"
    systemd-run --user --unit "$SERVICE_NAME" --collect \
        --setenv="PATH=$PATH" \
        --property=Type=exec --property="WorkingDirectory=$PROJECT_DIR" \
        --property="StandardOutput=append:$RUN_DIR/console.log" \
        --property="StandardError=append:$RUN_DIR/console.log" \
        "$PROJECT_DIR/.venv/bin/simex" "$RESEARCH_COMMAND" --run "$RUN_DIR" --config "$CONFIG_PATH"
    RESEARCH_PID="$(systemctl --user show "$SERVICE_NAME" --property=MainPID --value)"
    [[ "$RESEARCH_PID" =~ ^[1-9][0-9]*$ ]] || { echo "Service exited; inspect $RUN_DIR/console.log" >&2; exit 1; }
    printf '%s\n' "$SERVICE_NAME" > "$RUN_DIR/research.service"
else
    setsid nohup "$PROJECT_DIR/.venv/bin/simex" "$RESEARCH_COMMAND" --run "$RUN_DIR" --config "$CONFIG_PATH" \
        >> "$RUN_DIR/console.log" 2>&1 < /dev/null 9>&- &
    RESEARCH_PID=$!
fi
printf '%s\n' "$RESEARCH_PID" > "$PID_FILE.tmp"
mv "$PID_FILE.tmp" "$PID_FILE"
echo "Started $RESEARCH_COMMAND: PID $RESEARCH_PID; log: $RUN_DIR/console.log"
echo "Status: python3 scripts/status.py --run '$RUN_DIR'"
