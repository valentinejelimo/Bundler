#!/bin/bash
# Generic Harbor verifier entrypoint.
# Reads the v3 $TESTS_DIR/verifier.json (default /tests/verifier.json) and runs
# file-source deterministic and rubric verifiers. Writes:
#   $VERIFIER_DIR/reward.txt    Harbor compatibility numeric reward
#   $VERIFIER_DIR/reward.json   Harbor compatibility numeric reward map
#   $VERIFIER_DIR/verifier-report.json   complete verifier-owned result artifact
#
# Same script ships with every task; only verifier.json varies.

set -euo pipefail

TESTS_DIR="${TESTS_DIR:-/tests}"
VERIFIER_DIR="${VERIFIER_DIR:-/logs/verifier}"
UV_VERSION="${UV_VERSION:-0.9.17}"
export TESTS_DIR VERIFIER_DIR

mkdir -p "$VERIFIER_DIR"
CLEANUP_UV_VENV=0
CLEANUP_VERIFIER_VENV=0
UV_VENV_DIR=""
VENV_DIR=""

cleanup_temp_dirs() {
    if [ "$CLEANUP_VERIFIER_VENV" = "1" ] && [ -n "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
    fi
    if [ "$CLEANUP_UV_VENV" = "1" ] && [ -n "$UV_VENV_DIR" ]; then
        rm -rf "$UV_VENV_DIR"
    fi
}
trap cleanup_temp_dirs EXIT

write_infra_failure() {
    local message="$1"
    echo "0.0" > "$VERIFIER_DIR/reward.txt"
    python3 - "$VERIFIER_DIR" "$message" <<'PY'
import json
import sys
from pathlib import Path

verifier_dir = Path(sys.argv[1])
message = sys.argv[2]
payload = {
    "task_id": None,
    "reward": 0.0,
    "summary": {
        "total_verifiers": 0,
        "passed": 0,
        "failed": 0,
        "errored": 1,
        "total_weight": 1.0,
        "passed_weight": 0.0
    },
    "assertion_results": [],
    "error": message
}
(verifier_dir / "reward.json").write_text(json.dumps({"reward": 0.0}, indent=2), encoding="utf-8")
(verifier_dir / "verifier-report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

ensure_uv() {
    if [ -n "${UV_BIN:-}" ] && [ -x "$UV_BIN" ]; then
        return 0
    fi
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
        return 0
    fi

    UV_VENV_DIR="$(mktemp -d /tmp/harbor-uv-venv.XXXXXX)"
    CLEANUP_UV_VENV=1
    python3 -m venv "$UV_VENV_DIR" || {
        write_infra_failure "uv bootstrap venv creation failed"
        return 1
    }
    "$UV_VENV_DIR/bin/python" -m pip install --no-cache-dir "uv==$UV_VERSION" || {
        write_infra_failure "uv bootstrap install failed"
        return 1
    }
    UV_BIN="$UV_VENV_DIR/bin/uv"
}

requirements_hash() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$VERIFIER_REQUIREMENTS" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$VERIFIER_REQUIREMENTS" | awk '{print $1}'
    else
        python3 - "$VERIFIER_REQUIREMENTS" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
    fi
}

PYTHON_BIN="python3"
VERIFIER_REQUIREMENTS="$TESTS_DIR/verifier_lib/requirements.txt"
if [ -f "$VERIFIER_REQUIREMENTS" ]; then
    ensure_uv || exit 1
    REQ_HASH="$(requirements_hash)"
    if [ -n "${VERIFIER_VENV_DIR:-}" ]; then
        VENV_DIR="$VERIFIER_VENV_DIR"
    else
        VENV_DIR="$(mktemp -d /tmp/harbor-verifier-venv.XXXXXX)"
        CLEANUP_VERIFIER_VENV=1
    fi
    VENV_MARKER="$VENV_DIR/.requirements.sha256"
    PYTHON_BIN="$VENV_DIR/bin/python"
    if [ ! -x "$PYTHON_BIN" ] || [ ! -f "$VENV_MARKER" ] || [ "$(cat "$VENV_MARKER")" != "$REQ_HASH" ]; then
        rm -rf "$VENV_DIR"
        mkdir -p "$(dirname "$VENV_DIR")"
        python3 -m venv "$VENV_DIR" || {
            write_infra_failure "verifier dependency venv creation failed"
            exit 1
        }
        "$UV_BIN" pip install --python "$PYTHON_BIN" -r "$VERIFIER_REQUIREMENTS" || {
            write_infra_failure "verifier dependency install failed"
            exit 1
        }
        echo "$REQ_HASH" > "$VENV_MARKER"
    fi
fi

if [ ! -f "$TESTS_DIR/verifier.json" ]; then
    echo "ERROR: $TESTS_DIR/verifier.json missing"
    write_infra_failure "$TESTS_DIR/verifier.json missing"
    exit 1
fi

if "$PYTHON_BIN" "$TESTS_DIR/verifier_runtime.py" 2>&1 | tee "$VERIFIER_DIR/runtime.log"; then
    exit 0
else
    EXIT=$?
    exit $EXIT
fi
