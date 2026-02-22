#!/usr/bin/env python3
"""
Entry point for the Litigation Data Extractor (Programm 1).
Starts the Streamlit UI on http://localhost:8501
"""

import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).parent / "data_extractor" / "app.py"


def main():
    print("=" * 60)
    print("  ⚖️  Litigation Data Extractor")
    print("  Predictive Litigation Analytics — Programm 1")
    print("=" * 60)
    print(f"\nStarte UI unter: http://localhost:8501\n")
    print("Zum Beenden: Ctrl+C\n")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m", "streamlit",
                "run",
                str(APP_PATH),
                "--server.port", "8501",
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
        print("\n✅ Extractor beendet.")
    except FileNotFoundError:
        print("❌ Streamlit nicht gefunden. Bitte installieren:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
