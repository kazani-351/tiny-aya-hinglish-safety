#!/bin/bash
# CPU copy of run_gpu.sh (no GPU). Run a local Python script on a free Colab CPU VM with the HF token available, then tear down.
# Usage: ./run_gpu.sh script.py [session-name]
#   OUT=/content/out ./run_gpu.sh gen.py   -> also downloads that remote dir's files into ./outs/
#   UPLOAD=outs/x.jsonl ./run_gpu.sh s.py   -> uploads one data file to /content/x.jsonl before the script runs
# The token file is uploaded to the VM, read by the script, deleted there, and never printed.
# Cleanup (delete token on VM, verify, stop VM, verify) runs on normal exit, crash, Ctrl-C,
# kill and terminal close. Not `set -e`: cleanup must run even if a step fails.
# `colab exec` exits 0 even when the remote script raises, so every script must print
# RUN_OK as its last line; without it this runner exits 1.

SCRIPT="$1"
NAME="${2:-lexsi-$(date +%H%M%S)}"
TOKEN_FILE="$HOME/.config/hf.env"
COLAB="colab --auth=oauth2"
# colab exec defaults to a 30 s timeout, far too short for model loads and generation.
EXEC_TIMEOUT="${EXEC_TIMEOUT:-7200}"
STARTED=0
EXEC_LOG=$(mktemp -t run_gpu)

if [ -z "$SCRIPT" ] || [ ! -f "$SCRIPT" ]; then
  echo "usage: $0 script.py [session-name]" >&2; exit 2
fi
if [ -n "$UPLOAD" ] && [ ! -f "$UPLOAD" ]; then
  echo "FAIL: UPLOAD file $UPLOAD missing" >&2; exit 2
fi
if [ ! -f "$TOKEN_FILE" ]; then
  echo "FAIL: $TOKEN_FILE missing (see HANDOFF.md, Tooling)" >&2; exit 2
fi

cleanup() {
  rc=$?
  trap '' INT TERM HUP   # don't let a second Ctrl-C interrupt cleanup
  if [ "$STARTED" = 1 ]; then
    if [ -n "$OUT" ]; then
      mkdir -p outs
      for f in $($COLAB ls -s "$NAME" "$OUT" 2>/dev/null | awk '{print $NF}' | grep -v '/$'); do
        $COLAB download -s "$NAME" "$OUT/$f" "outs/$f" >/dev/null 2>&1 && echo "[run_gpu] downloaded outs/$f"
      done
    fi
    $COLAB rm -s "$NAME" /content/hf.env >/dev/null 2>&1
    if $COLAB ls -s "$NAME" /content 2>/dev/null | grep -q 'hf\.env'; then
      echo "[run_gpu] WARNING: token file still on VM (VM is being stopped anyway)"
    else
      echo "[run_gpu] token file not on VM"
    fi
    $COLAB stop -s "$NAME" >/dev/null 2>&1
    if $COLAB sessions 2>/dev/null | grep -q "$NAME"; then
      echo "[run_gpu] FAIL: session $NAME still active. Run: $COLAB stop -s $NAME" >&2
      rc=1
    else
      echo "[run_gpu] session $NAME stopped"
    fi
  fi
  rm -f "$EXEC_LOG"
  echo "[run_gpu] exit=$rc"
  exit $rc
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

$COLAB new -s "$NAME" || { echo "FAIL: could not get a CPU session" >&2; exit 1; }
STARTED=1
$COLAB upload -s "$NAME" "$TOKEN_FILE" /content/hf.env || { echo "FAIL: token upload" >&2; exit 1; }
# Positive control: proves the "token file not on VM" check in cleanup can see the file.
$COLAB ls -s "$NAME" /content 2>/dev/null | grep -q 'hf\.env' \
  || { echo "FAIL: uploaded token not visible via colab ls; cleanup check would be blind" >&2; exit 1; }
# Optional data file for the script (e.g. saved replies): UPLOAD=outs/x.jsonl -> /content/x.jsonl
if [ -n "$UPLOAD" ]; then
  $COLAB upload -s "$NAME" "$UPLOAD" "/content/$(basename "$UPLOAD")" || { echo "FAIL: data upload" >&2; exit 1; }
fi
$COLAB exec -s "$NAME" --timeout "$EXEC_TIMEOUT" -f "$SCRIPT" 2>&1 | tee "$EXEC_LOG"
if ! grep -a -q '^RUN_OK' "$EXEC_LOG"; then  # whole log: the colab client can print a traceback after RUN_OK
  echo "[run_gpu] FAIL: remote script did not finish (no RUN_OK line)" >&2
  exit 1
fi
