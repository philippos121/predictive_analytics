#!/usr/bin/env python3
"""
Entry point for the Predictive Litigation Analytics Model (Programm 2).
Starts the Streamlit UI on http://localhost:8502 (fallback: 8504)
"""

import socket
import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).parent / "model" / "app.py"
PREFERRED_PORT = 8502
FALLBACK_PORT = 8504


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def main():
    print("=" * 60)
    print("  Predictive Litigation Analytics")
    print("  ML Model Trainer & Predictor — Programm 2")
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
                "--theme.primaryColor", "#4a235a",
                "--theme.backgroundColor", "#ffffff",
                "--theme.secondaryBackgroundColor", "#f5f0fa",
                "--theme.textColor", "#1a365d",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nModell-App beendet.")
    except FileNotFoundError:
        print("Streamlit nicht gefunden. Bitte installieren:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
