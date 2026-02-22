#!/usr/bin/env python3
"""
Entry point for generating the PDF documentation.
Output: docs/PLA_Dokumentation.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("=" * 60)
    print("  📄 Predictive Litigation Analytics")
    print("  Dokumentations-Generator")
    print("=" * 60)

    try:
        from docs.generate_docs import generate_documentation
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        print("Bitte zuerst installieren: pip install -r requirements.txt")
        sys.exit(1)

    output = None
    if len(sys.argv) > 1:
        output = sys.argv[1]

    print("\nGeneriere PDF-Dokumentation...")
    path = generate_documentation(output)
    print(f"\n✅ Dokumentation gespeichert: {path}")
    print(f"   Dateigröße: {Path(path).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
