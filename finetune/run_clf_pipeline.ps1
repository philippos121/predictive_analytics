# run_clf_pipeline.ps1 — Classification-Head Fine-Tuning Pipeline
# Ausfuehren: .\run_clf_pipeline.ps1

Write-Host "=== Schritt 1: Daten vorbereiten ===" -ForegroundColor Cyan
python prepare_data_clf.py --input ../data/extracted/cases_dataset.json --output data/prepared_dataset_clf --val-ratio 0.1 --max-samples 3000
if ($LASTEXITCODE -ne 0) { Write-Host "Fehler bei Datenvorbereitung!" -ForegroundColor Red; exit 1 }

Write-Host "`n=== Schritt 2: Fine-Tuning mit Classification Head ===" -ForegroundColor Cyan
python finetune_clf.py --dataset data/prepared_dataset_clf --output output/legal-lora-clf
if ($LASTEXITCODE -ne 0) { Write-Host "Fehler beim Training!" -ForegroundColor Red; exit 1 }

Write-Host "`n=== Schritt 3: Inferenz (interaktiv) ===" -ForegroundColor Cyan
python inference_clf.py --adapter output/legal-lora-clf/final

Write-Host "`n=== Fertig! ===" -ForegroundColor Green
