# ⚖️ Predictive Litigation Analytics

Machine-Learning-System zur Vorhersage von Verfahrensausgängen bei österreichischen Zivilprozessen.

## Übersicht

Dieses System besteht aus zwei Programmen:

1. **Data Extractor** (`run_extractor.py`) — Strukturierte Datenerfassung aus PDF-Urteilen
2. **ML Model** (`run_model.py`) — Training, Evaluation und Vorhersage

## Schnellstart

```bash
bash install.sh
export OPENAI_API_KEY='sk-...'
python run_extractor.py   # → http://localhost:8501
python run_model.py       # → http://localhost:8502
python run_docs.py        # → docs/PLA_Dokumentation.pdf
```

## Dokumentation

Vollständige Dokumentation: `python run_docs.py` → `docs/PLA_Dokumentation.pdf`

## Technologie

- **KI-Extraktion**: OpenAI gpt-4o-mini
- **Embeddings**: text-embedding-3-large (3072 dim)
- **ML**: PyTorch (LitigationClassifier)
- **UI**: Streamlit
- **Daten**: JSON + HDF5 (lokal)
