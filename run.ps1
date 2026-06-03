# script to run the entire pipeline locally on windows power shell

# EXECUTION INSTRUCTIONS:
# To run this pipeline, execute the following 
# command in your Windows PowerShell terminal:
#
# .\run.ps1

$ErrorActionPreference = "Stop"

Write-Host " Starting Agentic Deep Research Pipeline" -ForegroundColor Cyan

Write-Host "`n[1/4] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`n[2/4] Running Phase 1: Scraping arXiv and building ChromaDB index..." -ForegroundColor Yellow
python src/phase1_index.py

Write-Host "`n[3/4] Running Phase 3: Agent Evaluation and Ablation Study..." -ForegroundColor Yellow
python src/phase3_eval.py

Write-Host "`n[4/4] Formatting submission files..." -ForegroundColor Yellow
python src/format_submission.py

Write-Host " Pipeline Complete!" -ForegroundColor Green
Write-Host " Formatted predictions are ready in the predictions/ directory." -ForegroundColor Green