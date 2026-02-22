#!/bin/bash
# Installation script for Predictive Litigation Analytics
# Usage: bash install.sh

set -e

echo "============================================================"
echo "  ⚖️  Predictive Litigation Analytics"
echo "  Installations-Skript"
echo "============================================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_MAJOR=3
REQUIRED_MINOR=10

echo "✓ Python Version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Erstelle virtuelle Umgebung..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate
echo "✓ Virtuelle Umgebung aktiviert"

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
echo "Installiere Abhängigkeiten (kann einige Minuten dauern)..."
pip install -r requirements.txt --quiet

echo ""
echo "✅ Installation abgeschlossen!"
echo ""
echo "============================================================"
echo "  Nächste Schritte:"
echo ""
echo "  1. API Key setzen:"
echo "     export OPENAI_API_KEY='sk-...'"
echo ""
echo "  2. Data Extractor starten:"
echo "     source venv/bin/activate"
echo "     python run_extractor.py"
echo "     → http://localhost:8501"
echo ""
echo "  3. ML Modell starten:"
echo "     source venv/bin/activate"
echo "     python run_model.py"
echo "     → http://localhost:8502"
echo ""
echo "  4. Dokumentation generieren:"
echo "     python run_docs.py"
echo "     → docs/PLA_Dokumentation.pdf"
echo "============================================================"
