# scripts/03_pinecone_upsert.py
"""
Uploads scene data to Pinecone vector database with embeddings.

Input: data_ready/*_upsert.jsonl (scene payloads ready for embedding)
Output: Data uploaded to Pinecone index 'convo'

This script:
1. Creates embeddings using OpenAI API
2. Ensures Pinecone index exists
3. Uploads vectors in batches to Pinecone

Required environment variables:
- OPENAI_API_KEY: OpenAI API key for embeddings
- PINECONE_API_KEY: Pinecone API key for vector database
"""

import os
import json
import time
import pathlib
from tqdm import tqdm
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

UP_DIR = "data_ready"
INDEX_NAME = "convo"                     # Pinecone index name
NAMESPACE = ""                           # Default namespace (empty)
EMBED_MODEL = "text-embedding-3-small"   # 1536-dimensional embeddings
BATCH_SIZE = 64                          # Batch size for processing


def get_clients():
    """
    Initialize OpenAI and Pinecone clients from environment variables.
    
    Returns:
        tuple: (OpenAI client, Pinecone client)
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY environment variable is required")
    
    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    
    return oai, pc


def ensure_index(pc):
    """
    Create Pinecone index if it doesn't exist and wait for it to be ready.
    """
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while True:
            desc = pc.describe_index(INDEX_NAME)
            if desc.status["ready"]:
                print(f"Index '{INDEX_NAME}' is ready!")
                break
            time.sleep(2)
    else:
        print(f"Index '{INDEX_NAME}' already exists")


def embed_batch(oai, texts):
    """
    Generate embeddings for a batch of texts using OpenAI API.
    
    Args:
        oai: OpenAI client
        texts (list): List of text strings to embed
        
    Returns:
        list: List of embedding vectors
    """
    try:
        resp = oai.embeddings.create(
            model=EMBED_MODEL, 
            input=texts
        )
        return [d.embedding for d in resp.data]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise


def upsert_file(oai, pc, path):
    """
    Process a single upsert file and upload to Pinecone.
    
    Args:
        oai: OpenAI client
        pc: Pinecone client
        path (str): Path to upsert JSONL file
    """
    index = pc.Index(INDEX_NAME)
    
    # Load all items from file
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    
    print(f"Processing {len(items)} scenes from {os.path.basename(path)}")
    
    # Process in batches
    for i in tqdm(range(0, len(items), BATCH_SIZE), desc="Uploading batches"):
        batch = items[i:i+BATCH_SIZE]
        texts = [item["text"] for item in batch]
        
        # Generate embeddings
        try:
            embeddings = embed_batch(oai, texts)
        except Exception as e:
            print(f"Failed to generate embeddings for batch {i//BATCH_SIZE + 1}: {e}")
            continue
        
        # Prepare vectors for upsert
        vectors = []
        for item, embedding in zip(batch, embeddings):
            vectors.append({
                "id": item["id"],
                "values": embedding,
                "metadata": item["metadata"]
            })
        
        # Upload to Pinecone
        try:
            index.upsert(vectors=vectors, namespace=NAMESPACE)
        except Exception as e:
            print(f"Failed to upsert batch {i//BATCH_SIZE + 1}: {e}")
            continue
    
    print(f"Successfully uploaded {len(items)} scenes from {os.path.basename(path)}")


def main():
    """Main function to process all upsert files."""
    if not os.path.exists(UP_DIR):
        print(f"Error: {UP_DIR} directory not found. Run previous scripts first.")
        return
    
    # Initialize clients
    try:
        oai, pc = get_clients()
        print("Successfully initialized OpenAI and Pinecone clients")
    except Exception as e:
        print(f"Failed to initialize clients: {e}")
        return
    
    # Ensure index exists
    try:
        ensure_index(pc)
    except Exception as e:
        print(f"Failed to create/access index: {e}")
        return
    
    # Process all upsert files
    upsert_files = [f for f in sorted(os.listdir(UP_DIR)) if f.endswith("_upsert.jsonl")]
    
    if not upsert_files:
        print("No upsert files found. Run 02_build_upsert_payload.py first.")
        return
    
    print(f"Found {len(upsert_files)} files to process")
    
    total_scenes = 0
    for fname in upsert_files:
        file_path = os.path.join(UP_DIR, fname)
        try:
            upsert_file(oai, pc, file_path)
            
            # Count scenes in file
            with open(file_path, "r") as f:
                scene_count = sum(1 for _ in f)
            total_scenes += scene_count
            
        except Exception as e:
            print(f"Failed to process {fname}: {e}")
            continue
    
    print(f"\n[DONE] Successfully uploaded {total_scenes} scenes to Pinecone index '{INDEX_NAME}'")


if __name__ == "__main__":
    main()