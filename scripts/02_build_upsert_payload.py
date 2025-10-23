# scripts/02_build_upsert_payload.py
"""
Converts parsed scene data into Pinecone upsert payloads.

Input: data_parsed/*_scenes.jsonl (structured scene data)
Output: data_ready/*_upsert.jsonl (ready for Pinecone embedding and upload)

This script transforms scene objects into the format required for Pinecone:
- id: unique scene identifier
- text: raw scene content for embedding
- metadata: all scene information for filtering and display
"""

import json
import os
import pathlib

PARSED_DIR = "data_parsed"
OUT_DIR = "data_ready"
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


def scene_to_upsert(scene):
    """
    Convert a scene object to Pinecone upsert format.
    
    Args:
        scene (dict): Scene object from parsing step
        
    Returns:
        dict: Upsert payload with id, text, and metadata
    """
    return {
        "id": scene["scene_id"],
        "text": scene["raw_text"],  # Raw text for embedding input
        "metadata": {
            "chunk_type": "scene",

            # Episode information
            "season": scene["season"],
            "episode_number": scene["episode_number"],
            "episode_id": scene["episode_id"],
            "episode_title": scene.get("episode_title"),

            # Scene information
            "scene_number": scene["scene_number"],
            "scene_id": scene["scene_id"],

            # Location and description
            "location": scene.get("location", ""),
            "scene_description": scene.get("scene_description", ""),

            # Characters present in scene
            "characters": scene.get("characters", []),

            # Database reference
            "firestore_path": scene["firestore_path"]
        }
    }


def main():
    """Main function to process all parsed scene files."""
    if not os.path.exists(PARSED_DIR):
        print(f"Error: {PARSED_DIR} directory not found. Run 01_parse_txt_to_scenes.py first.")
        return

    processed_files = 0
    total_scenes = 0

    for fname in sorted(os.listdir(PARSED_DIR)):
        if not fname.endswith("_scenes.jsonl"):
            continue

        print(f"Processing {fname}...")
        
        # Generate output filename
        out_path = os.path.join(OUT_DIR, fname.replace("_scenes", "_upsert"))
        scene_count = 0
        
        # Process each scene line
        with open(os.path.join(PARSED_DIR, fname), "r", encoding="utf-8") as r, \
             open(out_path, "w", encoding="utf-8") as w:
            
            for line in r:
                scene = json.loads(line)
                upsert_payload = scene_to_upsert(scene)
                w.write(json.dumps(upsert_payload, ensure_ascii=False) + "\n")
                scene_count += 1
        
        print(f"[ready] {fname} → {out_path} ({scene_count} scenes)")
        processed_files += 1
        total_scenes += scene_count

    print(f"\nSummary: {processed_files} files processed, {total_scenes} total scenes ready for upsert")


if __name__ == "__main__":
    main()