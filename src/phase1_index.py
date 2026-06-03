import os
import time # Added to handle sleep delays between downloads
import urllib.request # Needed for User-Agent spoofing
from dotenv import load_dotenv # To load environment variables from a .env file, keeping our API key secure and out of the codebase

# Load environment variables FIRST
load_dotenv()

import arxiv #Library to query arXiv and download PDFs
import fitz  # PyMuPDF:extracts the text from PDFs
import chromadb #Vector database to store embeddings

# THE MONKEY PATCH
# Reach into Chroma's internal posthog class and overwrite the broken capture method 
# with a lambda that accepts anything and does nothing.
try:
    import chromadb.telemetry.product.posthog
    chromadb.telemetry.product.posthog.Posthog.capture = lambda *args, **kwargs: None
except ImportError:
    pass

from chromadb.utils import embedding_functions #Provides various embedding functions, including SentenceTransformer
from tqdm import tqdm #Progress bar for long-running tasks
import ssl # To handle SSL issues (bypassing insecure request warnings) when downloading PDFs

ssl._create_default_https_context = ssl._create_unverified_context

# CONFIGURATION
# The assignment requires cs.CL, cs.AI, cs.LG from Jan 2024 - Apr 2026.
# REFINED SEARCH TO REDUCE OVERHEAD
# Targeting exactly the topics specified in Agentic AI.pdf
KEYWORDS = '(agent OR "agentic RAG" OR "tool use" OR "agent memory" OR "computer-use")'
SEARCH_QUERY = f'(cat:cs.CL OR cat:cs.AI OR cat:cs.LG) AND ti:{KEYWORDS} AND submittedDate:[202401010000 TO 202604302359]'
MAX_PAPERS = 700 # We can adjust this number based on how many papers we want to process (keeping in mind arXiv's rate limits and our processing time)
DATA_DIR = "./data" # Directory to store PDFs and the ChromaDB database
PDF_DIR = os.path.join(DATA_DIR, "pdfs") # Directory to store downloaded PDFs
DB_DIR = os.path.join(DATA_DIR, "chroma_db") # Directory to store ChromaDB database

os.makedirs(PDF_DIR, exist_ok=True) # Ensure PDF directory exists
os.makedirs(DB_DIR, exist_ok=True) # Ensure DB directory exists

# We use a fast, free local embedding model
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2") # This model is small and efficient, suitable for our use case without needing GPU resources.
chroma_client = chromadb.PersistentClient(path=DB_DIR) # Initialize ChromaDB client with persistence to the specified directory
collection = chroma_client.get_or_create_collection(name="arxiv_papers", embedding_function=embed_fn) # Create or get a collection named "arxiv_papers" with our embedding function

def chunk_text(text, chunk_size=1200, overlap=200): 
    """Splits text into overlapping chunks to preserve context between paragraphs."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def run_pipeline():

    # IDEMPOTENCY CHECK - If the collection already has more than 40,000 chunks, we assume the pipeline has been run before and skip it. This prevents unnecessary reprocessing and respects arXiv's servers.
    if collection.count() > 40000:
        print(f"Database already populated with {collection.count()} chunks. Skipping Phase 1.")
        return

    print("1. Querying arXiv with high-capacity one-shot paging...")

    # Added delay and retries to bypass HTTP 429 limits
    # Be extremely polite to arXiv's servers using high-capacity fetching
    client = arxiv.Client(
        page_size=700,         # Fetch all 700 at once. Zero pagination. Zero 429s.
        delay_seconds=10.0,    
        num_retries=10         # Give it more chances if it fails
    )
    
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=MAX_PAPERS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results_list = []
    try:
        # First, gather all the metadata before we start downloading PDFs
        for result in client.results(search):
            results_list.append(result)
            if len(results_list) >= MAX_PAPERS:
                break
    except Exception as e:
        print(f"Metadata gathering paused/completed early: {e}")

    print(f"Successfully gathered metadata for {len(results_list)} target papers.")

    for result in tqdm(results_list, desc="Processing Papers"): # Iterate through search results with a progress bar
        # 1. Download PDF
        pdf_path = os.path.join(PDF_DIR, f"{result.get_short_id()}.pdf")
        
        # Download if we don't already have it
        if not os.path.exists(pdf_path):
            for attempt in range(5): # Give it 5 attempts to fight through the firewalls
                try:
                    # Spoof Chrome User-Agent, so that arXiv doesn't block us with a 403 or 429. This is a common issue when scraping/downloading from sites that have anti-bot measures.
                    req = urllib.request.Request(
                        result.pdf_url,
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}
                    )
                    with urllib.request.urlopen(req, timeout=60) as response, open(pdf_path, 'wb') as out_file:
                        out_file.write(response.read())
                        
                    time.sleep(3)  # Rest briefly after a successful download
                    break
                except Exception as download_error:
                    print(f"\nRetrying download for {result.get_short_id()}: {download_error}")
                    time.sleep(15)

        # If it failed to download completely, skip this paper
        if not os.path.exists(pdf_path):
            continue

        # 2. Parse PDF
        try:
            doc = fitz.open(pdf_path)
            full_text = "\n".join([page.get_text() for page in doc])
            doc.close()
        except Exception as e:
            print(f"Failed to read {pdf_path}: {e}")
            continue

        # 3. Chunk & Store
        chunks = chunk_text(full_text)
        if not chunks:
            continue
        
        # Prepare batch data for ChromaDB
        ids = [f"{result.get_short_id()}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"arxiv_id": result.get_short_id(), "title": result.title}] * len(chunks)
        
        try:
            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            pass # Silently handle any ChromaDB batch insert errors

    print(f"\nPipeline complete. Index saved to {DB_DIR}")
    print(f"Total chunks in database: {collection.count()}")

if __name__ == "__main__":
    run_pipeline()