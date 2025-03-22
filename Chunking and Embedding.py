import chromadb
from chromadb.utils import embedding_functions  # Although not used directly, keep the import
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import logging
import os

# --- Configuration ---

# Configure logging
logging.basicConfig(
    filename='chroma_ingestion_log.txt',  # Change filename for clarity
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Data Loading ---

# Load the DataFrame
try:
    df = pd.read_csv('hadiths_data_with_question_and_answers_final.csv')
    logging.info("Successfully loaded 'hadiths_data_with_question_and_answers_final.csv'.")
except FileNotFoundError:
    logging.error(
        "Error: 'hadiths_data_with_question_and_answers_final.csv' not found.  Please ensure the file exists."
    )
    exit()
except Exception as e:
    logging.error(
        f"Error loading 'hadiths_data_with_question_and_answers_final.csv': {e}"
    )
    exit()


# --- ChromaDB Initialization and Collection Creation ---

# Initialize ChromaDB client
try:
    client = chromadb.PersistentClient(path="hadith_synthetic_rag_source")
    logging.info("ChromaDB client initialized.")
except Exception as e:
    logging.error(f"Error initializing ChromaDB client: {e}")
    exit()

# Check if the collection exists. Create if it doesn't, get it if it does.
collection_name = "hadith_synthetic_rag_source_complete"
try:
    collection = client.get_collection(name=collection_name)
    logging.info(f"ChromaDB collection '{collection_name}' loaded successfully.")
except ValueError:  # Collection doesn't exist
    logging.info(f"ChromaDB collection '{collection_name}' not found. Creating a new one.")
    collection = client.create_collection(name=collection_name)
except Exception as e:
    logging.error(f"Error accessing ChromaDB collection '{collection_name}': {e}")
    exit()

# --- Model Loading (Sentence Transformers with CUDA) ---

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    logging.info("SentenceTransformer model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading SentenceTransformer model: {e}")
    exit()

# --- Data Processing and Embedding Generation ---

try:
    # Create documents from the DataFrame, improve formatting
    documents = [
        f"""Rawi: {row['Rawi']}\nChapter: {row['Chapter']}\nReference: {row['Reference']}\nHadith Number: {row['Hadith Number']}\nNarator: {row['Narator']}\nHadith Text: {row['Hadiths Text']}\nSample Question: {row['Sample Question']}\nAnswer: {row['Synthetic Answer']}"""
        for _, row in df.iterrows()
    ]

    ids = [f"row_{i}" for i in range(len(documents))]

    # Compute embeddings using CUDA
    logging.info("Computing embeddings...")
    # Batch processing to avoid potential CUDA memory issues
    batch_size = 32  # Adjust batch size based on GPU memory. Lower for low memory GPU.
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch_documents = documents[i:i + batch_size]
        with torch.no_grad():  # Disable gradient calculation during inference
            batch_embeddings = model.encode(batch_documents, convert_to_tensor=True, device=device)
            embeddings.extend(batch_embeddings.cpu().numpy().tolist())  # Convert to list after moving to CPU

    logging.info(f"Embeddings computed for {len(embeddings)} documents.")


    # Add documents to the collection
    logging.info("Adding documents to ChromaDB...")
    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings
    )
    logging.info("Documents added to ChromaDB successfully.")

except KeyError as e:
    logging.error(
        f"Error: Missing column '{e}' in the DataFrame. Please check the column names."
    )
    exit()
except RuntimeError as e: # for CUDA out of memory
    logging.error(f"RuntimeError during embedding generation: {e}. Consider reducing batch size")
    exit()
except Exception as e:
    logging.error(f"An unexpected error occurred during data processing: {e}")
    exit()


# --- Debugging and Verification ---

try:
    # Debugging print
    num_docs = collection.count()
    print(f"Number of documents in collection: {num_docs}")
    logging.info(f"Number of documents in collection: {num_docs}")
except Exception as e:
    logging.error(f"Error counting documents in collection: {e}")

print("ChromaDB ingestion complete.")