#official grader script for the Agentic Deep Research Challenge for mac/linux environments. This script will run the entire pipeline from start to finish, including installing dependencies, building the vector database, running evaluations, and formatting the final output for submission. Make sure to run this script in a clean environment to avoid any conflicts with existing packages.
#!/bin/bash

# EXECUTION INSTRUCTIONS:
# To run this pipeline, execute the following 
# commands in your Linux/macOS terminal:
#
# chmod +x run.sh
# ./run.sh

# Exit immediately if a command exits with a non-zero status
set -e

echo " Starting Agentic Deep Research Pipeline"

# 1. Install dependencies
echo -e "\n[1/4] Installing dependencies..."
pip install -r requirements.txt

# 2. Build the Vector Database (Downloads arXiv PDFs and chunks them)
echo -e "\n[2/4] Running Phase 1: Scraping arXiv and building ChromaDB index..."
python src/phase1_index.py

# 3. Run the Evaluation/Ablation loops
echo -e "\n[3/4] Running Phase 3: Agent Evaluation and Ablation Study..."
echo "Note: This will read from eval/questions.jsonl and output to predictions/"
python src/phase3_eval.py

# 4. Format the final output to meet strict grader schemas
echo -e "\n[4/4] Formatting submission files..."
python src/format_submission.py

echo " Pipeline Complete!"
echo " Formatted predictions are ready in the predictions/ directory."