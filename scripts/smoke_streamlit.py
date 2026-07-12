from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
HEALTH_URL = "http://127.0.0.1:8765/_stcore/health"


def main() -> int:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            "--server.port=8765",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        for _ in range(30):
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                print(output)
                return process.returncode or 1

            try:
                with urlopen(HEALTH_URL, timeout=2) as response:
                    if response.read().decode("utf-8").strip() == "ok":
                        print("Streamlit health check passed.")
                        return 0
            except (URLError, TimeoutError):
                time.sleep(1)

        print("Streamlit did not become healthy within 30 seconds.")
        return 1
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
