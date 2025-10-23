# scripts/01_parse_txt_to_scenes.py
"""
Parses Friends episode transcripts from raw text files into structured scene data.

Input: data_raw/*.txt (one file = one episode)
Output: data_parsed/*_scenes.jsonl (one line = one scene)

This script extracts:
- Episode titles (THE ONE ... format)
- Scene boundaries and metadata
- Character dialogue and actions
- Scene descriptions and locations
"""

import re
import json
import os
import pathlib

RAW_DIR = "data_raw"
OUT_DIR = "data_parsed"
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Regex patterns for parsing
SCENE_RE = re.compile(
    r'^\[Scene:\s*(?P<loc>[^,\]]+)\s*,\s*(?P<desc>.+?)\]',
    re.IGNORECASE
)
DIALOGUE_RE = re.compile(r'^([A-Za-z][A-Za-z ]*):\s*(.+)')  # Allow spaces in names (e.g., "Gunther Jr")
EP_TITLE_RE = re.compile(r'^\s*(THE ONE .+)')               # Episode title pattern


def parse_episode_id(fname):
    """
    Extract episode information from filename.
    
    Args:
        fname (str): Filename like 'S01E01.txt'
        
    Returns:
        tuple: (episode_id, season, episode_number)
    """
    stem = pathlib.Path(fname).stem
    season = int(stem[1:3])
    episode = int(stem[4:6])
    return stem, season, episode


def parse_file(path):
    """
    Parse a single episode transcript file into structured scenes.
    
    Args:
        path (str): Path to the episode text file
        
    Returns:
        list: List of scene dictionaries
    """
    episode_id, season, episode_number = parse_episode_id(os.path.basename(path))
    
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # 1) Find episode title (first THE ONE ... line)
    episode_title = None
    for ln in lines:
        m = EP_TITLE_RE.match(ln)
        if m:
            episode_title = m.group(1).strip()
            break

    scenes = []
    scene_buffer = []
    scene_number = 0
    location_current = None
    scene_desc_current = None

    def flush_scene():
        """Process and save the current scene buffer."""
        nonlocal scene_number, scene_buffer, location_current, scene_desc_current
        if not scene_buffer:
            return

        scene_number += 1
        scene_id = f"{episode_id}_{scene_number:03d}"
        raw_text = "\n".join(scene_buffer).strip()

        # Parse lines and collect characters
        structured, chars, idx = [], set(), 0
        for ln in scene_buffer:
            idx += 1
            dm = DIALOGUE_RE.match(ln)
            if dm:
                spk, txt = dm.group(1).strip(), dm.group(2)
                chars.add(spk)
                structured.append({
                    "line_number": idx, 
                    "type": "dialogue", 
                    "speaker": spk, 
                    "text": txt
                })
            elif ln.startswith("(") and ln.endswith(")"):
                structured.append({
                    "line_number": idx, 
                    "type": "action", 
                    "text": ln
                })
            else:
                structured.append({
                    "line_number": idx, 
                    "type": "narration", 
                    "text": ln
                })

        scene = {
            "episode_id": episode_id,
            "season": season,
            "episode_number": episode_number,
            "episode_title": episode_title,

            "scene_number": scene_number,
            "scene_id": scene_id,

            "location": (location_current or "").strip(),
            "scene_description": (scene_desc_current or "").strip(),

            "characters": sorted(chars, key=str),  # Preserve original case
            "firestore_path": f"/episodes/{episode_id}/scenes/{scene_id}",

            "raw_text": raw_text,
            "lines": structured
        }
        scenes.append(scene)

        # Reset for next scene
        scene_buffer = []
        location_current = None
        scene_desc_current = None

    # Process all lines
    for ln in lines:
        m = SCENE_RE.match(ln)
        if m:
            # New scene starts - flush previous scene
            flush_scene()
            location_current = m.group("loc")
            scene_desc_current = m.group("desc")
            scene_buffer.append(ln)  # Include scene header in raw text
        else:
            scene_buffer.append(ln)

    # Don't forget the last scene
    flush_scene()
    return scenes


def main():
    """Main function to process all episode files."""
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.lower().endswith(".txt"):
            continue
            
        print(f"Processing {fname}...")
        scenes = parse_file(os.path.join(RAW_DIR, fname))
        
        # Write scenes to JSONL file
        out_path = os.path.join(OUT_DIR, f"{os.path.splitext(fname)[0]}_scenes.jsonl")
        with open(out_path, "w", encoding="utf-8") as w:
            for sc in scenes:
                w.write(json.dumps(sc, ensure_ascii=False) + "\n")
        
        print(f"[parsed] {fname}: {len(scenes)} scenes → {out_path}")


if __name__ == "__main__":
    main()