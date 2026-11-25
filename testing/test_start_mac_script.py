import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "start_mac.sh"


def test_startup_script_defines_required_setup_and_services():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "command -v python3" in script
    assert "python3 -m venv" in script
    assert 'source "${VENV_DIR}/bin/activate"' in script
    assert "python -m pip install -r" in script
    assert ".env.example" in script
    assert "backend.main:app" in script and "--port 8000" in script
    assert "streamlit run frontend/app.py" in script and "--server.port 8501" in script
    assert "trap cleanup INT TERM EXIT" in script


@pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason="Signal and process-group behavior requires Bash on macOS/Linux.",
)
def test_startup_script_creates_venv_runs_services_and_stops_them(
    tmp_path,
):
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(SCRIPT_PATH, scripts / "start_mac.sh")
    (project / "requirements.txt").write_text("", encoding="utf-8")
    (project / ".env.example").write_text("", encoding="utf-8")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$STARTUP_TEST_LOG"
case "$*" in
    *"-m uvicorn "*) exec sleep 60 ;;
    *"-m streamlit "*) exec sleep 60 ;;
esac
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_python3 = fake_bin / "python3"
    fake_python3.write_text(
        """#!/usr/bin/env bash
if [[ "$1" != "-m" || "$2" != "venv" ]]; then
    exit 1
fi
mkdir -p "$3/bin"
cp "$FAKE_VENV_PYTHON" "$3/bin/python"
cat > "$3/bin/activate" <<'ACTIVATE'
VIRTUAL_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VIRTUAL_ENV
PATH="$VIRTUAL_ENV/bin:$PATH"
export PATH
ACTIVATE
""",
        encoding="utf-8",
    )
    fake_python3.chmod(0o755)

    log_path = tmp_path / "commands.log"
    venv_path = tmp_path / "test-venv"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(fake_bin) + os.pathsep + environment.get("PATH", ""),
            "DRIFTING_ORACLE_VENV_DIR": str(venv_path),
            "FAKE_VENV_PYTHON": str(fake_python),
            "STARTUP_TEST_LOG": str(log_path),
        }
    )
    bash = shutil.which("bash")
    assert bash is not None
    process = subprocess.Popen(
        [bash, str(scripts / "start_mac.sh")],
        cwd=project,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if log_path.exists():
                logged = log_path.read_text(encoding="utf-8")
                if "-m uvicorn " in logged and "-m streamlit " in logged:
                    break
            if process.poll() is not None:
                break
            time.sleep(0.05)

        assert process.poll() is None
        assert venv_path.joinpath("bin", "activate").is_file()
        logged = log_path.read_text(encoding="utf-8")
        assert "-m pip install -r" in logged
        assert "-m uvicorn backend.main:app" in logged
        assert "-m streamlit run frontend/app.py" in logged

        os.killpg(process.pid, signal.SIGINT)
        output, _ = process.communicate(timeout=10)
        assert "No .env file found" in output
        assert "http://127.0.0.1:8501" in output
        assert "http://127.0.0.1:8000/docs" in output
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate(timeout=10)
