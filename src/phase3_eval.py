import json # For reading and writing JSONL files
import os # For file path management and checking if files exist
import time # For adding delays between API calls to avoid rate limits
import sys # To modify the Python path for imports

# Add the src directory to the Python path so we can import our phase2 module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importing this automatically runs the telemetry monkey patch and connects to Chroma/Gemini!
from phase2_agent import ResearchAgent, model, collection

# Define our 5 Ablation Configurations
CONFIGURATIONS = {
    "baseline": {"use_planner": False, "use_reflector": False, "use_verifier": False},
    "full_agent": {"use_planner": True, "use_reflector": True, "use_verifier": True},
    "no_planner": {"use_planner": False, "use_reflector": True, "use_verifier": True},
    "no_reflector": {"use_planner": True, "use_reflector": False, "use_verifier": True},
    "no_verifier": {"use_planner": True, "use_reflector": True, "use_verifier": False}
}

# PATHS
QUESTIONS_FILE = "./eval/questions.jsonl"
PREDICTIONS_DIR = "./predictions"

os.makedirs(PREDICTIONS_DIR, exist_ok=True) # Ensure the predictions directory exists so we can save our results there

def load_questions(filepath):
    """Loads the assignment questions from the JSONL file."""
    questions = []
    with open(filepath, 'r') as f: # Read the questions file line by line (since it's JSONL) and parse each line as JSON, appending it to the questions list
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions

def run_evaluation():
    if not os.path.exists(QUESTIONS_FILE):
        print(f"ERROR: Could not find {QUESTIONS_FILE}. Please make sure you placed the assignment file there!")
        return

    questions = load_questions(QUESTIONS_FILE)
    print(f"Loaded {len(questions)} questions for evaluation.")

    # Loop through each of the 5 configurations
    for config_name, flags in CONFIGURATIONS.items():
        output_file = os.path.join(PREDICTIONS_DIR, f"{config_name}.jsonl")
        
        results = []
        start_index = 0
        
        # RESUME LOGIC: If the output file already exists, read it and determine how many questions have already been answered. This allows us to resume from where we left off in case of interruptions or API limits.
        if os.path.exists(output_file):
            # Read what we already have
            with open(output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            
            start_index = len(results)
            
            if start_index >= len(questions):
                print(f"\n✅ Skipping '{config_name}' (100% complete).")
                continue
            else:
                print(f"\n⚠️ Resuming '{config_name}' from Question {start_index + 1}...")

        print(f" STARTING ABLATION RUN: {config_name.upper()}")
        print(f" Flags: {flags}")
        
        # Initialize the agent with the specific ablation flags
        agent = ResearchAgent(
            model=model, 
            collection=collection, 
            max_steps=3,
            use_planner=flags["use_planner"],
            use_reflector=flags["use_reflector"],
            use_verifier=flags["use_verifier"]
        )

        # Start from where we left off
        for index in range(start_index, len(questions)):
            q = questions[index]
            print(f"\n[Question {index + 1}/{len(questions)}]")
            question_id = q.get("_id", str(index)) 
            question_text = q.get("question", "")
            
            # Run the agent!
            try:
                answer = agent.run(question_text)
            except Exception as e:
                print(f"CRITICAL ERROR on question {question_id}: {e}")
                answer = "Error generating answer due to API limits or system crash."
                time.sleep(10)
            
            # Save the specific format required by the grader
            results.append({
                "_id": question_id,
                "answer": answer
            })
            
            # Write EVERYTHING to file immediately (Overwrites safely with updated list)
            with open(output_file, 'w') as f:
                for res in results:
                    f.write(json.dumps(res) + '\n')
            
            # Polite API spacing between questions
            time.sleep(8) # Increased to 8 seconds to give the API more breathing room

        print(f"\n✅ Finished '{config_name}'. Results saved to {output_file}")

if __name__ == "__main__":
    run_evaluation()