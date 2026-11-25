#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${DRIFTING_ORACLE_VENV_DIR:-${PROJECT_ROOT}/.venv}"
cd "${PROJECT_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required. Install Python 3 and rerun this script." >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "The virtual environment is incomplete: ${VENV_DIR}" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
    echo "No .env file found. Copy .env.example to .env and configure its values."
    echo "Admin sign-in requires bcrypt hashes and a signing secret; see README.md."
fi

echo "Dashboard: http://127.0.0.1:8501"
echo "API docs:  http://127.0.0.1:8000/docs"
echo "Press Ctrl+C to stop both services."

api_pid=""
streamlit_pid=""

cleanup() {
    trap - INT TERM EXIT
    local pid
    for pid in "${api_pid}" "${streamlit_pid}"; do
        if [[ -n "${pid}" ]]; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done
    for pid in "${api_pid}" "${streamlit_pid}"; do
        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null || true
        fi
    done
}

trap cleanup INT TERM EXIT
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
api_pid=$!
python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501 &
streamlit_pid=$!

wait "${api_pid}" "${streamlit_pid}"
