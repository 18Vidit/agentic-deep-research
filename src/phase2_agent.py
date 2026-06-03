import os
import time

from dotenv import load_dotenv # To load environment variables from a .env file, keeping our API key secure and out of the codebase

# Load environment variables FIRST, before importing Chroma
# Load environment variables (API Key and other secrets) from a .env file. 
load_dotenv() # This will look for a .env file in the current directory and load the variables into the environment

import chromadb

# THE MONKEY PATCH
# Reach into Chroma's internal posthog class and overwrite the broken capture method 
# with a lambda that accepts anything and does nothing.
try:
    import chromadb.telemetry.product.posthog
    chromadb.telemetry.product.posthog.Posthog.capture = lambda *args, **kwargs: None
except ImportError:
    pass

from chromadb.utils import embedding_functions # Provides various embedding functions, including SentenceTransformer
import google.generativeai as genai # Gemini API client library

genai.configure(api_key=os.environ["GEMINI_API_KEY"]) # Configure the Gemini API client with our API key from the environment variable

# Initialize the Gemini Model (Flash is extremely fast, perfect for agentic loops)[might change to Gemini-2 if we want more reasoning power, but Flash should be sufficient for retrieval and simple reasoning tasks]
# We use system instructions to force it to return clean text without markdown formatting when needed
model = genai.GenerativeModel('gemini-flash-lite-latest')

# Connect to our existing local ChromaDB
DATA_DIR = "./data" # Directory to store PDFs and the ChromaDB database
DB_DIR = os.path.join(DATA_DIR, "chroma_db") # Directory where our ChromaDB database is stored (created in phase1_index.py)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2") # We need to use the same embedding function as in phase1_index.py to ensure compatibility when querying the database

try: # Test connection to ChromaDB and print the number of documents in the collection
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(name="arxiv_papers", embedding_function=embed_fn)
    print(f"Successfully connected to ChromaDB. Found {collection.count()} chunks.")
except Exception as e: # Handle exceptions that may occur during connection (e.g., database not found, corrupted database, etc.)
    print(f"Error connecting to ChromaDB: {e}")

def retrieve_documents(query: str, n_results: int = 3): 
    """
    The tool our agent will use to search the database.
    Takes a search string, embeds it, and returns the top chunks.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )# The results include 'documents' (the text chunks) and 'metadatas' (which contain the arXiv IDs and other info we stored)
    
    # Format the results so the LLM can easily read them
    formatted_results = []
    for i in range(len(results['documents'][0])): # Iterate through the returned documents (chunks) and their corresponding metadata
        doc_text = results['documents'][0][i]
        arxiv_id = results['metadatas'][0][i]['arxiv_id']
        formatted_results.append(f"[Paper: {arxiv_id}]\n{doc_text}\n")
        
    return formatted_results

class ResearchAgent:
    '''
    This class encapsulates the entire agentic loop, including planning, retrieval, reflection, and synthesis.
    The agent will use the Gemini LLM for all its reasoning and generation tasks, and it will use the retrieve_documents function to get information from our ChromaDB when needed.
    '''
    def __init__(self, model, collection, max_steps=3, use_planner=True, use_reflector=True, use_verifier=True):
        self.model = model
        self.collection = collection
        # If we have no reflector, we only ever do 1 step
        self.max_steps = max_steps if use_reflector else 1 
        
        # ABLATION FEATURE FLAGS
        self.use_planner = use_planner
        self.use_reflector = use_reflector
        self.use_verifier = use_verifier

    def _call_llm(self, prompt):
        """A resilient wrapper that automatically handles API rate limits."""
        for attempt in range(5): # Try up to 5 times
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e: # Catch any exception (including rate limits, timeouts, etc.)
                if "429" in str(e) or "ResourceExhausted" in str(e) or "quota" in str(e).lower():
                    wait_time = 10 * (attempt + 1) # Wait 10s, then 20s, then 30s...
                    # Print the exact error so we aren't flying blind
                    error_msg = str(e).splitlines()[0][:100] 
                    print(f"    [API Rate Limit Hit] {error_msg}...")
                    print(f"    Pausing for {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise e # If it's a different error, crash normally
        
        # If it fails 5 times, try one last time and let it crash if it fails
        return self.model.generate_content(prompt).text.strip()

    def _planner(self, question):
        """Phase A: The agent decides what specific search query it needs."""
        prompt = f"""You are a research planner. The user wants to know: "{question}"
        Write a single, highly specific search query to find this in an academic database.
        Return ONLY the search query, no extra text."""
        return self._call_llm(prompt)

    def _reflector(self, question, gathered_evidence):
        """Phase B: The agent critiques its own research so far."""
        prompt = f"""You are a harsh research critic. 
        Question: "{question}"
        Evidence gathered so far: 
        {gathered_evidence}
        
        Does the evidence contain enough specific facts to answer the question completely?
        Reply with exactly 'YES' or 'NO'."""
        return 'YES' in self._call_llm(prompt).upper()

    def _synthesizer(self, question, gathered_evidence):
        """Phase C: The agent writes the final answer with citations."""
        prompt = f"""You are a research assistant.
        Answer the question STRICTLY using the provided evidence.
        
        Question: "{question}"
        Evidence: {gathered_evidence}
        
        Rules:
        1. Base your answer ONLY on the evidence. Do not use outside knowledge.
        2. When you state a fact from a paper, cite it inline like this: [1234.56789].
        
        Write the final answer now:"""
        return self._call_llm(prompt)

    def _citation_verifier(self, draft_answer, gathered_evidence):
        """Phase D: The agent checks its own citations against the text."""
        prompt = f"""You are a strict peer reviewer.
        
        Evidence available: {gathered_evidence}
        Draft Answer: {draft_answer}
        
        Task:
        1. Read the draft answer.
        2. Check every inline citation (e.g., [1234.56789]).
        3. If a claim is NOT explicitly supported by the evidence from that specific paper, remove the citation.
        4. Rewrite the answer with only the verified citations.
        
        Return ONLY the rewritten, verified answer."""
        return self._call_llm(prompt)

    def run(self, question):
        """The core loop, upgraded with Ablation Feature Flags."""
        print(f"\n[Agent Started] Task: {question}")
        gathered_evidence = ""
        
        for step in range(self.max_steps):
            print(f"\ Step {step + 1}")
            
            # FEATURE FLAG: PLANNER
            if self.use_planner:
                search_query = self._planner(question)
                print(f"Planner: '{search_query}'")
            else:
                search_query = question
                print(f"Bypassing Planner. Using raw query: '{search_query}'")
                
            time.sleep(3) # Rate limit pause
            
            new_docs = retrieve_documents(search_query, n_results=3)
            new_evidence = "".join(new_docs)
            print(f"Retriever: Found {len(new_docs)} chunks.")
            
            gathered_evidence += f"\nSearch Query: {search_query}\nResults:\n{new_evidence}\n"
            
            # FEATURE FLAG: REFLECTOR
            if self.use_reflector:
                is_sufficient = self._reflector(question, gathered_evidence)
                if is_sufficient:
                    print("Reflector: Evidence sufficient. Breaking loop.")
                    break
                else:
                    print("Reflector: Evidence insufficient. Searching again...")
                    time.sleep(3)
            else:
                print("Bypassing Reflector. Moving directly to synthesis.")
                break # Exit the loop immediately after one retrieval
                
        print("\ Final Synthesis")
        if not gathered_evidence.strip():
            return "No evidence found."
            
        time.sleep(3) 
        draft_answer = self._synthesizer(question, gathered_evidence)
        print("Draft generated.")
        
        # FEATURE FLAG: VERIFIER
        if self.use_verifier:
            print("Verifying citations...")
            time.sleep(3) 
            final_answer = self._citation_verifier(draft_answer, gathered_evidence)
        else:
            print("Bypassing Verifier. Using raw draft.")
            final_answer = draft_answer
            
        print("\n[Final Answer]")
        print(final_answer)
        
        return final_answer

if __name__ == "__main__":
    # Test the Agent
    test_question = "What is the SWE-agent, and what does it do?"
    
    agent = ResearchAgent(model=model, collection=collection, max_steps=3)
    agent.run(test_question)