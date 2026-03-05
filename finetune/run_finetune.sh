#!/usr/bin/env bash
# =============================================================================
# run_finetune.sh — Kompletter Ablauf: Daten vorbereiten → Finetuning → Test
#
# Voraussetzungen:
#   - NVIDIA GPU mit mind. 16 GB VRAM
#   - CUDA 12.1+ installiert
#   - Python 3.10+
#   - Hugging Face Account (für Modell-Download): huggingface-cli login
#
# Verwendung:
#   bash finetune/run_finetune.sh <pfad_zur_json_datei>
#
# Beispiel mit Testdaten:
#   cd finetune
#   python generate_sample_data.py -n 10000 -o sample_cases.json
#   bash run_finetune.sh sample_cases.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Farben
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== LLM-Finetuning für Verfahrensausgang-Prognose ===${NC}"
echo ""

# --- 0. Argumente prüfen ---
DEFAULT_INPUT="../data/extracted/cases_dataset.json"
INPUT_JSON="${1:-$DEFAULT_INPUT}"

if [ ! -f "$INPUT_JSON" ]; then
    echo -e "${RED}Fehler: Datei '$INPUT_JSON' nicht gefunden.${NC}"
    echo ""
    echo -e "${YELLOW}Verwendung: bash run_finetune.sh [json_datei]${NC}"
    echo "  Ohne Argument wird '$DEFAULT_INPUT' verwendet."
    echo ""
    echo "Testdaten erzeugen:"
    echo "  python generate_sample_data.py -n 10000 -o sample_cases.json"
    echo "  bash run_finetune.sh sample_cases.json"
    exit 1
fi

echo "  Eingabedatei: $INPUT_JSON"

# --- 1. Abhängigkeiten installieren ---
echo -e "${YELLOW}[1/4] Installiere Abhängigkeiten...${NC}"
pip install -q -r requirements_finetune.txt

# --- 2. GPU prüfen ---
echo -e "${YELLOW}[2/4] Prüfe GPU...${NC}"
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA nicht verfügbar!'
gpu = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f'  GPU: {gpu} ({mem:.1f} GB VRAM)')
if mem < 14:
    print(f'  WARNUNG: Nur {mem:.1f} GB VRAM. Mind. 16 GB empfohlen.')
"

# --- 3. Daten vorbereiten ---
echo -e "${YELLOW}[3/4] Bereite Trainingsdaten vor...${NC}"
python prepare_data.py \
    --input "$INPUT_JSON" \
    --output data/prepared_dataset \
    --val-ratio 0.1

# --- 4. Finetuning starten ---
echo -e "${YELLOW}[4/4] Starte QLoRA-Finetuning...${NC}"
echo "  Modell: LeoLM/leo-mistral-hessianai-7b-chat"
echo "  Methode: QLoRA (4-bit NF4 + LoRA r=64)"
echo ""

python finetune.py \
    --model LeoLM/leo-mistral-hessianai-7b-chat \
    --dataset data/prepared_dataset \
    --output output/legal-lora

echo ""
echo -e "${GREEN}=== Finetuning abgeschlossen! ===${NC}"
echo ""
echo "Nächste Schritte:"
echo "  # Interaktive Prognose:"
echo "  python inference.py --adapter output/legal-lora/final"
echo ""
echo "  # Batch-Prognose:"
echo "  python inference.py --adapter output/legal-lora/final --batch test_cases.json"
