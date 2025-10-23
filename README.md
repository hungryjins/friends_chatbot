# Friends Chatbot Data Pipeline

A comprehensive data processing pipeline for Friends TV show transcripts, designed to parse episode scripts into structured scene data and upload them to Pinecone vector database for semantic search and chatbot applications.

## 📁 Project Structure

```
project/
├─ data_raw/                      # Original episode .txt files (one file = one episode)
│   ├─ S01E01.txt
│   ├─ S01E02.txt
│   └─ ...
├─ data_parsed/                   # Scene-level JSONL files (intermediate output)
│   ├─ S01E01_scenes.jsonl
│   └─ ...
├─ data_ready/                    # Upsert payload JSONL files (ready for Pinecone)
│   ├─ S01E01_upsert.jsonl
│   ├─ plots_upsert.jsonl            # Episode plot summaries for Pinecone
│   └─ ...
├─ scripts/
│   ├─ 01_parse_txt_to_scenes.py     # Parse raw transcripts into scenes
│   ├─ 02_build_upsert_payload.py    # Build Pinecone upsert payloads
│   ├─ 03_pinecone_upsert.py         # Upload to Pinecone with embeddings
│   ├─ 04_parse_plots_pdf.py         # Parse episode plots from PDF guide
│   └─ 05_pinecone_plots_upsert.py   # Upload plot summaries to Pinecone
├─ requirements.txt               # Python dependencies
├─ .env.template                  # Environment variables template
└─ README.md                      # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
cd Friends_chatbot

# Install Python dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.template .env
# Edit .env file with your API keys
```

### 2. Configure API Keys

Edit the `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

**Getting API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Pinecone: https://app.pinecone.io/

### 3. Prepare Data

Place your Friends episode transcript files in the `data_raw/` directory:
- Format: `S01E01.txt`, `S01E02.txt`, etc.
- One file per episode
- Files should contain the full episode transcript

### 4. Run the Pipeline

Execute the scripts in order:

```bash
# Step 1: Parse transcripts into structured scenes
python scripts/01_parse_txt_to_scenes.py

# Step 2: Build Pinecone upsert payloads
python scripts/02_build_upsert_payload.py

# Step 3: Upload to Pinecone with embeddings
python scripts/03_pinecone_upsert.py

# Optional: Add episode plot summaries
# Step 4: Parse episode plots from PDF guide
python scripts/04_parse_plots_pdf.py

# Step 5: Upload plot summaries to Pinecone
python scripts/05_pinecone_plots_upsert.py
```

## 📋 Detailed Documentation

### Data Processing Specification

#### Input Format Requirements

**Episode Files:**
- Must be named in format: `S{season:02d}E{episode:02d}.txt`
- Example: `S01E01.txt`, `S10E18.txt`
- Encoding: UTF-8

**Content Structure:**
- Episode title: First line containing `THE ONE ...`
- Scene headers: `[Scene: <location>, <scene description>]`
- Dialogue: `Speaker: text`
- Actions: `(action description)`
- Other text: Treated as narration

#### Output Schemas

**Scene Object (data_parsed/*.jsonl):**
```json
{
  "episode_id": "S01E01",
  "season": 1,
  "episode_number": 1,
  "episode_title": "THE ONE WHERE MONICA GETS A NEW ROOMMATE (THE PILOT-THE UNCUT VERSION)",
  
  "scene_number": 1,
  "scene_id": "S01E01_001",
  
  "location": "Central Perk",
  "scene_description": "Chandler, Joey, Phoebe, and Monica are there.",
  
  "characters": ["Monica","Rachel","Ross","Chandler","Joey","Phoebe"],
  "firestore_path": "/episodes/S01E01/scenes/S01E01_001",
  
  "raw_text": "[Scene: Central Perk, Chandler, Joey, Phoebe, and Monica are there.]\\nMonica: There's nothing to tell! ...\\n...",
  
  "lines": [
    {"line_number": 1, "type": "narration", "text": "[Scene: Central Perk, Chandler, Joey, Phoebe, and Monica are there.]"},
    {"line_number": 2, "type": "dialogue", "speaker": "Monica", "text": "There's nothing to tell! ..."},
    {"line_number": 3, "type": "action", "text": "(They all stare, bemused.)"}
  ]
}
```

**Upsert Payload (data_ready/*.jsonl):**
```json
{
  "id": "S01E01_001",
  "text": "[Scene: Central Perk, ...]\\nMonica: There's nothing to tell! ...",
  "metadata": {
    "chunk_type": "scene",
    "season": 1,
    "episode_number": 1,
    "episode_id": "S01E01",
    "episode_title": "THE ONE WHERE MONICA GETS A NEW ROOMMATE...",
    "scene_number": 1,
    "scene_id": "S01E01_001",
    "location": "Central Perk",
    "scene_description": "Chandler, Joey, Phoebe, and Monica are there.",
    "characters": ["Monica","Rachel","Ross","Chandler","Joey","Phoebe"],
    "firestore_path": "/episodes/S01E01/scenes/S01E01_001"
  }
}
```

**Plot Payload (data_ready/plots_upsert.jsonl):**
```json
{
  "id": "S01E01_plot",
  "text": "Friends Episode S01E01: Pilot - Rachel leaves Barry at the alter and moves in with Monica. Monica goes on a date with Paul the wine guy, who turns out to be less than sincere. Ross is depressed about his failed marriage. Joey compares women to ice cream. Everyone watches Spanish soaps. Ross reveals his high school crush on Rachel.",
  "metadata": {
    "doc_type": "plot",
    "season": 1,
    "episode_number": 1,
    "episode_id": "S01E01",
    "episode_title": "Pilot",
    "plot_text": "Rachel leaves Barry at the alter and moves in with Monica. Monica goes on a date with Paul the wine guy, who turns out to be less than sincere. Ross is depressed about his failed marriage. Joey compares women to ice cream. Everyone watches Spanish soaps. Ross reveals his high school crush on Rachel.",
    "chunk_type": "plot",
    "word_count": 54
  }
}
```

### Script Details

#### 01_parse_txt_to_scenes.py

**Purpose:** Converts raw episode transcripts into structured scene data.

**Key Features:**
- Extracts episode titles from `THE ONE ...` format
- Identifies scene boundaries using `[Scene: ...]` markers
- Parses dialogue, actions, and narration
- Maintains original text formatting and character names
- Generates unique scene IDs with zero-padded numbering

**Processing Rules:**
- Scene headers: Must match pattern `[Scene: <location>, <description>]`
- Dialogue: Format `Speaker: text` (supports spaces in names)
- Actions: Text within parentheses `(action)`
- Everything else: Treated as narration
- Character names: Preserved with original capitalization

#### 02_build_upsert_payload.py

**Purpose:** Transforms parsed scenes into Pinecone-ready format.

**Key Features:**
- Converts scene objects to upsert payloads
- Separates embedding text from metadata
- Maintains all scene information for filtering
- Generates Firestore-compatible paths

#### 03_pinecone_upsert.py

**Purpose:** Uploads scenes to Pinecone vector database with embeddings.

**Key Features:**
- Creates OpenAI embeddings using `text-embedding-3-small` (1536-dim)
- Manages Pinecone index creation and configuration
- Batch processing for efficient uploads (64 items per batch)
- Progress tracking with tqdm
- Error handling and retry logic
- Uses index name `convo` as specified

**Configuration:**
- Index: `convo`
- Embedding Model: `text-embedding-3-small`
- Dimensions: 1536
- Metric: Cosine similarity
- Cloud: AWS (us-east-1)

#### 04_parse_plots_pdf.py

**Purpose:** Extracts episode plot summaries from Friends Guide PDF.

**Key Features:**
- Parses PDF containing all episode plot summaries
- Extracts season/episode numbers and titles
- Cleans and structures plot text
- Validates episode counts (236 total episodes across 10 seasons)
- Outputs structured plot data

**Processing Rules:**
- Identifies episodes by "Season X Episode Y" pattern
- Extracts episode titles from "The One..." format
- Parses plot summaries between title and next episode
- Handles special characters and formatting

#### 05_pinecone_plots_upsert.py

**Purpose:** Uploads episode plot summaries to Pinecone vector database.

**Key Features:**
- Creates embeddings for plot summaries using OpenAI
- Uploads to same `convo` index as scene data
- Uses `chunk_type: "plot"` to distinguish from scene data
- Batch processing for efficient uploads
- Includes word count and metadata for filtering

**Data Structure:**
- Each plot summary becomes one vector
- ID format: `S01E01_plot`
- Separate from scene data but in same index
- Enables plot-based search and recommendations

## 🔧 Advanced Usage

### Custom Configuration

You can modify the following constants in the scripts:

**01_parse_txt_to_scenes.py:**
- `RAW_DIR`: Input directory for episode files
- `OUT_DIR`: Output directory for parsed scenes
- Regular expressions for parsing patterns

**03_pinecone_upsert.py:**
- `INDEX_NAME`: Pinecone index name (default: "convo")
- `EMBED_MODEL`: OpenAI embedding model
- `BATCH_SIZE`: Batch size for processing (default: 64)

**04_parse_plots_pdf.py:**
- `PDF_PATH`: Path to Friends Guide PDF file
- `OUT_DIR`: Output directory for plot data

**05_pinecone_plots_upsert.py:**
- `INDEX_NAME`: Pinecone index name (same as scenes: "convo")
- `EMBED_MODEL`: OpenAI embedding model
- `BATCH_SIZE`: Batch size for plot uploads

### Monitoring and Debugging

All scripts include comprehensive logging and progress tracking:

```bash
# Example output
Processing S01E01.txt...
[parsed] S01E01.txt: 15 scenes → data_parsed/S01E01_scenes.jsonl

Processing S01E01_scenes.jsonl...
[ready] S01E01_scenes.jsonl → data_ready/S01E01_upsert.jsonl (15 scenes)

Successfully initialized OpenAI and Pinecone clients
Index 'convo' already exists
Found 1 files to process
Processing 15 scenes from S01E01_upsert.jsonl
Uploading batches: 100%|████████████| 1/1 [00:02<00:00,  2.34s/it]

Processing 236 plot summaries...
Uploading plot batches: 100%|████████████| 4/4 [00:08<00:00,  2.12s/it]
Successfully uploaded 236 plot summaries
```

### Error Handling

The pipeline includes robust error handling:

- **Missing files:** Scripts check for required input directories
- **API failures:** Automatic retry logic for network issues
- **Invalid data:** Graceful handling of malformed input
- **Environment setup:** Clear error messages for missing API keys

## 📊 Data Insights

### Parsing Statistics

The pipeline provides insights about your data:

- Total episodes processed
- Scenes per episode
- Characters identified
- Processing time and performance

### Scene Structure

Each scene includes:
- **Location tracking:** Where the scene takes place
- **Character presence:** Who appears in each scene
- **Content types:** Mix of dialogue, actions, and narration
- **Hierarchical organization:** Season → Episode → Scene

## 🤝 Integration

### Using with Chatbots

The processed data is ready for chatbot integration:

1. **Semantic Search:** Query Pinecone for relevant scenes or plot summaries
2. **Context Retrieval:** Use metadata for filtering by character, location, season
3. **Dual Search Types:** 
   - Scene search: Find specific dialogue and interactions
   - Plot search: Find episodes by storyline and summary
4. **Response Generation:** Feed retrieved scenes to language models

### Example Query Flow

```python
# Query for specific scenes
scene_results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
    filter={"chunk_type": "scene", "characters": {"$in": ["Monica", "Rachel"]}}
)

# Query for plot summaries
plot_results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True,
    filter={"chunk_type": "plot", "season": {"$gte": 1, "$lte": 3}}
)

# Use retrieved content as context for chatbot response
context_scenes = [match["metadata"]["text"] for match in scene_results["matches"]]
context_plots = [match["metadata"]["plot_text"] for match in plot_results["matches"]]
```

## 📝 Requirements

### Python Dependencies

- `openai>=1.0.0` - OpenAI API client for embeddings
- `pinecone-client>=3.0.0` - Pinecone vector database client
- `tqdm>=4.64.0` - Progress bars for batch processing

### API Requirements

- **OpenAI API:** For generating text embeddings
- **Pinecone:** For vector storage and similarity search

### System Requirements

- Python 3.8+
- Stable internet connection for API calls
- Sufficient disk space for intermediate files

## 🔍 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**API key errors:**
- Verify your API keys are correct in `.env`
- Check API key permissions and billing status

**Memory issues with large datasets:**
- Reduce `BATCH_SIZE` in `03_pinecone_upsert.py`
- Process episodes in smaller batches

**Parsing errors:**
- Check episode file encoding (should be UTF-8)
- Verify scene header format: `[Scene: location, description]`

### Performance Optimization

**For large datasets:**
- Use parallel processing for multiple episodes
- Implement chunked processing for memory efficiency
- Cache embeddings to avoid reprocessing

**For faster uploads:**
- Increase batch size (if memory allows)
- Use multiple API keys with rate limiting
- Consider Pinecone's bulk upload features

## 📚 Additional Resources

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Friends Episode Transcripts](https://www.livesinabox.com/friends/scripts.html)

## 🎯 Next Steps

After running the pipeline:

1. **Validate Data:** Check Pinecone dashboard for uploaded vectors
2. **Test Queries:** Run sample similarity searches
3. **Build Chatbot:** Integrate with your conversation system
4. **Monitor Performance:** Track query response times and accuracy

## 📄 License

This project is for educational and research purposes. Please respect copyright laws when using TV show transcripts.