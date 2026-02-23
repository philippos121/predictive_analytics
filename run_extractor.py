#!/usr/bin/env python3
"""
Entry point for the Litigation Data Extractor (Programm 1).
Starts the Streamlit UI on http://localhost:8501 (fallback: 8503)
"""

import socket
import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).parent / "data_extractor" / "app.py"
PREFERRED_PORT = 8501
FALLBACK_PORT = 8503


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def main():
    print("=" * 60)
    print("  Litigation Data Extractor")
    print("  Predictive Litigation Analytics — Programm 1")
    print("=" * 60)

    port = PREFERRED_PORT if _port_free(PREFERRED_PORT) else FALLBACK_PORT
    print(f"\nStarte UI unter: http://localhost:{port}\n")
    print("Zum Beenden: Ctrl+C\n")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m", "streamlit",
                "run",
                str(APP_PATH),
                "--server.port", str(port),
                "--server.headless", "false",
                "--browser.gatherUsageStats", "false",
                "--theme.primaryColor", "#2d6a9f",
                "--theme.backgroundColor", "#ffffff",
                "--theme.secondaryBackgroundColor", "#f0f4f8",
                "--theme.textColor", "#1a365d",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nExtractor beendet.")
    except FileNotFoundError:
        print("Streamlit nicht gefunden. Bitte installieren:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
