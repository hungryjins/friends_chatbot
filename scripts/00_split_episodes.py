# scripts/00_split_episodes.py
"""
Split Friends master transcript file into individual episode files using "THE ONE" pattern.

Input: Friends_Transcript.txt (master transcript file)  
Output: data_raw/S01E01.txt, S01E02.txt, etc. (individual episode files)

Finds episodes by looking for lines containing "THE ONE" (case insensitive).
Validates each episode title against episode_titles.json for manual verification.
"""

import os
import re
import pathlib
import json
from difflib import SequenceMatcher

RAW_MASTER = "Friends_Transcript.txt"
OUT_DIR = "data_raw"
TITLES_JSON = "episode_titles.json"

# Actual episode counts per season (Friends)
SEASON_EPISODES = [24, 24, 25, 24, 24, 25, 24, 24, 24, 18]  # Total: 236

# Pattern to find episode titles (case insensitive, may have numbers at start)
EPISODE_PATTERN = re.compile(r'^\s*(\d+\s*[:\-→]?\s*)?THE ONE\b', re.IGNORECASE)


def idx_to_season_episode(idx: int):
    """
    Convert 1-based episode index to (season, episode_number).
    
    Args:
        idx (int): 1-based episode index
        
    Returns:
        tuple: (season, episode_number)
        
    Examples:
        idx=1 -> (1,1), idx=24 -> (1,24), idx=25 -> (2,1), ...
    """
    if idx < 1:
        raise ValueError("Episode index must be >= 1")
    
    remaining = idx
    for season, count in enumerate(SEASON_EPISODES, start=1):
        if remaining <= count:
            return season, remaining
        remaining -= count
    
    # Index exceeds total episodes
    raise ValueError(f"Episode index {idx} exceeds total episodes {sum(SEASON_EPISODES)}")


def similarity_score(str1, str2):
    """Calculate similarity between two strings (0.0 to 1.0)"""
    # Normalize strings: remove punctuation, convert to lowercase, remove extra spaces
    def normalize(s):
        s = re.sub(r'[^\w\s]', '', s.lower())
        return ' '.join(s.split())
    
    norm1 = normalize(str1)
    norm2 = normalize(str2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def validate_title_match(actual_title, expected_title, ep_id, similarity_threshold=0.7):
    """
    Validate if actual title matches expected title.
    Returns (is_match, similarity_score, needs_manual_check)
    """
    score = similarity_score(actual_title, expected_title)
    
    if score >= 0.95:  # Very high match
        return True, score, False
    elif score >= similarity_threshold:  # Good match but show warning
        return True, score, True
    else:  # Poor match - needs manual verification
        return False, score, True


def write_episode(ep_idx, buffer_lines, expected_titles):
    """
    Write episode buffer to SxxExx.txt file with title validation.
    
    Args:
        ep_idx (int): Episode index (1-based)
        buffer_lines (list): Lines of the episode
        expected_titles (dict): Dictionary of expected titles
        
    Returns:
        tuple: (output_file_path, needs_manual_check)
    """
    if ep_idx < 1 or not buffer_lines:
        return None, False

    season, ep_num = idx_to_season_episode(ep_idx)
    ep_id = f"S{season:02d}E{ep_num:02d}"
    out_path = os.path.join(OUT_DIR, f"{ep_id}.txt")
    text = "\n".join(buffer_lines).strip()

    # Prevent empty files
    if not text:
        return None, False

    # Extract actual title from first line
    actual_title = buffer_lines[0].strip() if buffer_lines else ""
    expected_title = expected_titles.get(ep_id, "Unknown")
    
    # Validate title match
    is_match, score, needs_check = validate_title_match(actual_title, expected_title, ep_id)
    
    # Write file
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        w.write(text)

    # Print results
    status_icon = "✅" if is_match and not needs_check else "⚠️" if is_match else "❌"
    print(f"[{status_icon}] {ep_id} → {out_path} ({len(buffer_lines)} lines)")
    print(f"    Actual:   {actual_title}")
    print(f"    Expected: {expected_title}")
    print(f"    Similarity: {score:.2f}")
    
    if needs_check:
        if is_match:
            print(f"    📝 WARNING: Moderate match - please verify manually")
        else:
            print(f"    🔍 MANUAL CHECK REQUIRED: Poor title match")
    print()

    return out_path, needs_check


def main():
    """Main function to split the master transcript file using 'THE ONE' pattern."""
    if not os.path.exists(RAW_MASTER):
        raise FileNotFoundError(f"Master file not found: {RAW_MASTER}")

    # Load episode titles for verification
    episode_titles = {}
    if os.path.exists(TITLES_JSON):
        with open(TITLES_JSON, 'r', encoding='utf-8') as f:
            episode_titles = json.load(f)
        print(f"✅ Loaded {len(episode_titles)} expected episode titles from {TITLES_JSON}")
    else:
        print(f"⚠️  Warning: {TITLES_JSON} not found - skipping title validation")

    print(f"📖 Reading master file: {RAW_MASTER}")
    with open(RAW_MASTER, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    episode_idx = 0               # 1-based counter
    buffer = []                   # Current episode line buffer
    found_episodes = []           # List of found episode titles
    manual_checks_needed = []     # Episodes needing manual verification

    print("🔍 Splitting episodes using 'THE ONE' pattern (case insensitive)...")
    print("=" * 80)
    
    for i, ln in enumerate(lines):
        # Check if this line contains "THE ONE"
        if EPISODE_PATTERN.match(ln):
            # Found new episode title
            if buffer:
                # Save previous episode
                out_file, needs_check = write_episode(episode_idx, buffer, episode_titles)
                if needs_check and out_file:
                    season, ep_num = idx_to_season_episode(episode_idx)
                    ep_id = f"S{season:02d}E{ep_num:02d}"
                    manual_checks_needed.append(ep_id)
                    
            # Start new episode
            episode_idx += 1
            found_episodes.append(ln.strip())
            buffer = [ln]
        else:
            # Continue adding to current episode buffer
            buffer.append(ln)

    # Save last episode
    if buffer:
        out_file, needs_check = write_episode(episode_idx, buffer, episode_titles)
        if needs_check and out_file:
            season, ep_num = idx_to_season_episode(episode_idx)
            ep_id = f"S{season:02d}E{ep_num:02d}"
            manual_checks_needed.append(ep_id)

    # Summary report
    total_expected = sum(SEASON_EPISODES)
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"'THE ONE' episodes found: {len(found_episodes)}")
    print(f"Episode files created: {episode_idx}")
    print(f"Friends total episodes (expected): {total_expected}")
    
    if manual_checks_needed:
        print(f"\n🔍 MANUAL VERIFICATION NEEDED ({len(manual_checks_needed)} episodes):")
        for ep_id in manual_checks_needed:
            print(f"   - {ep_id}")
        print("\n📝 Please manually verify these episodes have correct content.")
    
    perfect_matches = episode_idx - len(manual_checks_needed)
    print(f"\n✅ Perfect matches: {perfect_matches}")
    print(f"⚠️  Need verification: {len(manual_checks_needed)}")
    
    if episode_idx != total_expected:
        print("ℹ️  INFO: Split episode count differs from official total. "
              "Master transcript may be incomplete or contain extra content.")
    
    print(f"\n🎉 Episode splitting complete! Check {OUT_DIR}/ directory.")
    
    # Show first few found episodes for verification
    print(f"\n📝 First 10 episode titles found:")
    for i, title in enumerate(found_episodes[:10], 1):
        print(f"   {i:2d}. {title}")
    
    if len(found_episodes) > 10:
        print(f"   ... and {len(found_episodes) - 10} more")


if __name__ == "__main__":
    main()