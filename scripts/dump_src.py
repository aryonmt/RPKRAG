#!/usr/bin/env python3
"""
Dumps the entire src/document_qa directory into a single text file.
Usage: uv run python scripts/dump_src.py
Output: src_dump.txt (in the root directory)
"""

import os
from pathlib import Path

# Define paths
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src" / "document_qa"
OUTPUT_FILE = ROOT / "src_dump.txt"

def dump_src():
    if not SRC_DIR.exists():
        print(f"❌ Error: Directory {SRC_DIR} does not exist.")
        return

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("PROJECT SOURCE CODE DUMP (src/document_qa)")
    output_lines.append("=" * 80)
    output_lines.append("")

    # Walk through the directory
    for root, dirs, files in os.walk(SRC_DIR):
        # Exclude __pycache__ and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        # Sort files for consistent output
        for file in sorted(files):
            if not file.endswith(".py"):
                continue
                
            file_path = Path(root) / file
            rel_path = file_path.relative_to(ROOT)
            
            output_lines.append(f"===== FILE: {rel_path} =====")
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    output_lines.append(content)
            except Exception as e:
                output_lines.append(f"[Error reading file: {e}]")
                
            output_lines.append("") # Empty line between files
            output_lines.append("") # Another empty line for spacing

    # Write to output file
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"✅ Success! Source code dumped to: {OUTPUT_FILE}")
        print(f"Total files processed: {len(output_lines) // 4}") # Rough estimate
    except Exception as e:
        print(f"❌ Error writing output file: {e}")

if __name__ == "__main__":
    dump_src()