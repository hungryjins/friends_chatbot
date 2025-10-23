# scripts/05_pinecone_plots_upsert.py
"""
Friends Plot Pinecone Upsert - Upload plot summaries to Pinecone vector database

Input: data_ready/plots_upsert.jsonl (236 plot summaries)
Output: Pinecone index with embedded plot data

This script:
- Reads plot payload data
- Generates OpenAI embeddings for plot summaries
- Uploads vectors to Pinecone with metadata
"""

import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Configuration
UP_DIR = "data_ready"
INDEX_NAME = "convo"
NAMESPACE = ""
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64
MAX_RETRIES = 3
RETRY_DELAY = 5


def get_clients():
    """Initialize OpenAI and Pinecone clients"""
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    if not openai_key or not pinecone_key:
        raise ValueError("OPENAI_API_KEY and PINECONE_API_KEY are required in .env file")
    
    print("🔑 Initializing clients...")
    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    
    return oai, pc


def embed_batch(oai, texts, retry_count=0):
    """Generate embeddings for a batch of texts with retry logic"""
    try:
        resp = oai.embeddings.create(
            model=EMBED_MODEL,
            input=texts
        )
        return [d.embedding for d in resp.data]
    
    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"  ⚠️ Embedding retry {retry_count + 1}/{MAX_RETRIES}: {e}")
            time.sleep(RETRY_DELAY * (retry_count + 1))
            return embed_batch(oai, texts, retry_count + 1)
        else:
            print(f"  ❌ Embedding failed after {MAX_RETRIES} retries: {e}")
            raise


def upsert_batch_to_pinecone(index, vectors, retry_count=0):
    """Upsert a batch of vectors to Pinecone with retry logic"""
    try:
        index.upsert(vectors=vectors, namespace=NAMESPACE)
        return True
    
    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"  ⚠️ Upsert retry {retry_count + 1}/{MAX_RETRIES}: {e}")
            time.sleep(RETRY_DELAY * (retry_count + 1))
            return upsert_batch_to_pinecone(index, vectors, retry_count + 1)
        else:
            print(f"  ❌ Upsert failed after {MAX_RETRIES} retries: {e}")
            return False


def upsert_plots_file(oai, pc):
    """Process and upsert plot data to Pinecone"""
    index = pc.Index(INDEX_NAME)
    plots_file = os.path.join(UP_DIR, "plots_upsert.jsonl")
    
    if not os.path.exists(plots_file):
        print(f"❌ File not found: {plots_file}")
        return False
    
    # Load all plot items
    print(f"📖 Loading plot data from {plots_file}...")
    items = []
    with open(plots_file, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    
    print(f"✅ Loaded {len(items)} plot summaries")
    
    # Process in batches
    successful_batches = 0
    failed_batches = 0
    total_vectors = 0
    
    for i in tqdm(range(0, len(items), BATCH_SIZE), desc="Processing plot batches"):
        batch = items[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        # Extract texts for embedding
        texts = [item["text"] for item in batch]
        
        print(f"\n📦 Batch {batch_num}/{(len(items) + BATCH_SIZE - 1) // BATCH_SIZE}")
        print(f"   → Generating embeddings for {len(texts)} items...")
        
        # Generate embeddings
        try:
            embeddings = embed_batch(oai, texts)
            print(f"   ✅ Embeddings generated successfully")
        except Exception as e:
            print(f"   ❌ Failed to generate embeddings: {e}")
            failed_batches += 1
            continue
        
        # Prepare vectors for Pinecone
        vectors = []
        for item, embedding in zip(batch, embeddings):
            vectors.append({
                "id": item["id"],
                "values": embedding,
                "metadata": item["metadata"]
            })
        
        print(f"   → Uploading {len(vectors)} vectors to Pinecone...")
        
        # Upload to Pinecone
        if upsert_batch_to_pinecone(index, vectors):
            print(f"   ✅ Batch uploaded successfully")
            successful_batches += 1
            total_vectors += len(vectors)
        else:
            print(f"   ❌ Failed to upload batch")
            failed_batches += 1
        
        # Small delay between batches
        time.sleep(1)
    
    # Summary
    print(f"\n📊 Upload Summary:")
    print(f"   ✅ Successful batches: {successful_batches}")
    print(f"   ❌ Failed batches: {failed_batches}")
    print(f"   📦 Total vectors uploaded: {total_vectors}")
    
    return failed_batches == 0


def verify_upload(pc):
    """Verify the upload by checking index stats and running sample queries"""
    print(f"\n🔍 Verifying upload...")
    
    try:
        index = pc.Index(INDEX_NAME)
        
        # Get index statistics
        stats = index.describe_index_stats()
        print(f"📊 Index Statistics:")
        print(f"   → Total vectors: {stats.total_vector_count}")
        print(f"   → Index dimension: {stats.dimension}")
        
        if hasattr(stats, 'namespaces') and NAMESPACE in stats.namespaces:
            ns_stats = stats.namespaces[NAMESPACE]
            print(f"   → Namespace '{NAMESPACE}' vectors: {ns_stats.vector_count}")
        
        # Sample query test
        print(f"\n🔍 Running sample queries...")
        
        # Query by specific plot ID
        sample_queries = [
            "S01E01_plot",  # First episode
            "S10E18_plot",  # Last episode
            "S05E12_plot"   # Random middle episode
        ]
        
        for query_id in sample_queries:
            try:
                result = index.query(
                    id=query_id,
                    top_k=3,
                    include_metadata=True,
                    namespace=NAMESPACE
                )
                
                if result.matches:
                    match = result.matches[0]
                    title = match.metadata.get('episode_title', 'N/A')
                    season = match.metadata.get('season', 'N/A')
                    print(f"   ✅ {query_id}: {title} (Season {season})")
                else:
                    print(f"   ❌ {query_id}: No matches found")
                    
            except Exception as e:
                print(f"   ❌ Query {query_id} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    """Main execution function"""
    print("🎬 Friends Plot Pinecone Upsert Starting")
    
    try:
        # Initialize clients
        oai, pc = get_clients()
        print("✅ Clients initialized successfully")
        
        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            print(f"❌ Index '{INDEX_NAME}' not found")
            print(f"Available indexes: {existing_indexes}")
            return
        
        print(f"✅ Using existing index: {INDEX_NAME}")
        
        # Process and upload plot data
        print(f"\n🔄 Starting plot data upload...")
        success = upsert_plots_file(oai, pc)
        
        if success:
            print(f"\n🎉 All plot data uploaded successfully!")
            
            # Verify upload
            verify_upload(pc)
            
            print(f"\n✨ Pipeline completed successfully!")
            print(f"   → 236 Friends episode plots are now searchable in Pinecone")
            print(f"   → You can now search for episodes by plot content")
            
        else:
            print(f"\n❌ Some uploads failed. Check logs above for details.")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()