# scripts/04_parse_plots_pdf.py
"""
Friends Plot PDF Parser - Parse Friends episode plot summaries from PDF

Input: Friends-Guide.pdf
Output: data_ready/plots_upsert.jsonl

Parses all 236 episodes (Seasons 1-10) plot summaries for Pinecone upsert data generation
"""

import re
import json
from typing import Dict, List, Tuple
import fitz  # PyMuPDF
import os
import pathlib


class FriendsPlotPDFParser:
    """Friends plot summary PDF parser"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
        # Expected episode count per season (for validation)
        self.expected_counts = {
            1: 24, 2: 24, 3: 25, 4: 24, 5: 24,
            6: 25, 7: 24, 8: 24, 9: 24, 10: 18
        }
        
        # Regex patterns (adapted to PDF structure)
        self.season_pattern = re.compile(r'(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)\s+Season\s+Plots', re.IGNORECASE)
        self.episode_pattern = re.compile(r'(\d+)\.(\d+)\s+(.+?)(?=\n)', re.IGNORECASE)
        
        # Season name to number mapping
        self.season_words = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10
        }
    
    def extract_text_from_pdf(self) -> str:
        """Extract text content from PDF file"""
        try:
            doc = fitz.open(self.pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"PDF reading error: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
        return text.strip()
    
    def parse_episode_block(self, block: str, season: int) -> List[Dict]:
        """Parse episode block (adapted to PDF structure)"""
        episodes = []
        
        # Split episodes by pattern (1.01 Title format)
        episode_matches = list(self.episode_pattern.finditer(block))
        
        for i, match in enumerate(episode_matches):
            season_num = int(match.group(1))  # First number in 1.01
            episode_num = int(match.group(2))  # Second number in 1.01
            title = match.group(3).strip()
            
            # Skip if season number doesn't match
            if season_num != season:
                continue
            
            # Extract text until next episode
            start_pos = match.end()
            if i + 1 < len(episode_matches):
                end_pos = episode_matches[i + 1].start()
                episode_text = block[start_pos:end_pos]
            else:
                episode_text = block[start_pos:]
            
            # Clean plot text (lines after title)
            plot_text = self.clean_text(episode_text)
            
            if plot_text:  # Only if plot text is not empty
                episode_id = f"S{season:02d}E{episode_num:02d}"
                
                episode_data = {
                    "season": season,
                    "episode_number": episode_num,
                    "episode_id": episode_id,
                    "title": title,
                    "plot_text": plot_text,
                    "word_count": len(plot_text.split())
                }
                episodes.append(episode_data)
        
        return episodes
    
    def parse_all_episodes(self, text: str) -> List[Dict]:
        """Parse all episodes (adapted to PDF structure)"""
        all_episodes = []
        
        # Split by seasons
        season_splits = self.season_pattern.split(text)
        
        # Skip first element (text before seasons)
        for i in range(1, len(season_splits), 2):
            if i + 1 < len(season_splits):
                season_name = season_splits[i].lower()  # "First", "Second", etc.
                season_text = season_splits[i + 1]
                
                # Convert season name to number
                season_num = self.season_words.get(season_name)
                if not season_num:
                    continue
                
                print(f"Parsing Season {season_num} ({season_name.title()})...")
                
                season_episodes = self.parse_episode_block(season_text, season_num)
                all_episodes.extend(season_episodes)
                
                print(f"  → {len(season_episodes)} episodes parsed")
        
        return all_episodes
    
    def validate_episodes(self, episodes: List[Dict]) -> bool:
        """Validate episode data quality"""
        season_counts = {}
        for ep in episodes:
            season = ep['season']
            season_counts[season] = season_counts.get(season, 0) + 1
        
        print("\n📊 Episode count validation by season:")
        total_expected = 0
        total_found = 0
        all_valid = True
        
        for season in range(1, 11):
            expected = self.expected_counts.get(season, 0)
            found = season_counts.get(season, 0)
            
            total_expected += expected
            total_found += found
            
            status = "✅" if found == expected else "❌"
            print(f"  Season {season}: {found}/{expected} {status}")
            
            if found != expected:
                all_valid = False
        
        print(f"\nTotal: {total_found}/{total_expected} {'✅' if all_valid else '❌'}")
        return all_valid
    
    def create_pinecone_payloads(self, episodes: List[Dict]) -> List[Dict]:
        """Create Pinecone upsert payloads"""
        payloads = []
        
        for ep in episodes:
            # Embedding text (title + plot)
            embedding_text = f"Friends Episode {ep['episode_id']}: {ep['title']} - {ep['plot_text']}"
            
            payload = {
                "id": f"{ep['episode_id']}_plot",
                "text": embedding_text,
                "metadata": {
                    "doc_type": "plot",
                    "season": ep["season"],
                    "episode_number": ep["episode_number"],
                    "episode_id": ep["episode_id"],
                    "episode_title": ep["title"],
                    "plot_text": ep["plot_text"],
                    "chunk_type": "plot",
                    "word_count": ep["word_count"]
                }
            }
            payloads.append(payload)
        
        return payloads
    
    def save_payloads(self, payloads: List[Dict], output_path: str):
        """Save payloads to JSONL file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for payload in payloads:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        
        print(f"✅ Payloads saved successfully: {output_path}")
        print(f"   → {len(payloads)} items")


def main():
    """Main execution function"""
    pdf_path = "./Friends-Guide.pdf"
    output_path = "data_ready/plots_upsert.jsonl"
    
    print("🎬 Friends Plot PDF Parser Starting")
    print(f"📁 Input file: {pdf_path}")
    print(f"📁 Output file: {output_path}")
    
    # Check PDF file exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    
    # Initialize parser
    parser = FriendsPlotPDFParser(pdf_path)
    
    # Step 1: Extract PDF text
    print("\n📖 Step 1: Extracting PDF text...")
    text = parser.extract_text_from_pdf()
    if not text:
        print("❌ PDF text extraction failed")
        return None
    print(f"✅ Text extraction completed ({len(text):,} characters)")
    
    # Step 2: Parse episodes
    print("\n🔍 Step 2: Parsing episodes...")
    episodes = parser.parse_all_episodes(text)
    print(f"✅ Parsing completed: {len(episodes)} episodes")
    
    # Step 3: Validate data
    print("\n✅ Step 3: Validating data...")
    is_valid = parser.validate_episodes(episodes)
    if not is_valid:
        print("⚠️ Some episodes are missing but continuing...")
    
    # Step 4: Create Pinecone payloads
    print("\n📦 Step 4: Creating Pinecone payloads...")
    payloads = parser.create_pinecone_payloads(episodes)
    print(f"✅ Payload creation completed: {len(payloads)} items")
    
    # Step 5: Save file
    print(f"\n💾 Step 5: Saving file...")
    parser.save_payloads(payloads, output_path)
    
    print("\n🎉 Parsing completed!")
    print(f"   → Total {len(episodes)} episode plots ready")
    print(f"   → Next step: python scripts/05_pinecone_plots_upsert.py")
    
    return episodes


if __name__ == "__main__":
    main()