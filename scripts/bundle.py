"""
Bundle the agent for Kaggle submission.

Creates a tar.gz archive with main.py at the root and the src/ package.
This is the file you submit via:
    kaggle competitions submit kaggriculture -f submission.tar.gz -m "description"

Usage:
    python scripts/bundle.py                    # creates submission.tar.gz
    python scripts/bundle.py -o my_agent.tar.gz # custom output name
"""

import os
import sys
import tarfile
import argparse
from datetime import datetime


def bundle(output_path: str = "submission.tar.gz"):
    """Create submission tar.gz with main.py and src/."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Files to include
    files_to_include = [
        ("agent/main.py", "main.py"),  # main.py must be at root
    ]

    # Include all .py files from src/
    src_dir = os.path.join(project_root, "src")
    for filename in sorted(os.listdir(src_dir)):
        if filename.endswith(".py"):
            files_to_include.append(
                (f"src/{filename}", f"src/{filename}")
            )

    output_full = os.path.join(project_root, output_path)

    with tarfile.open(output_full, "w:gz") as tar:
        for src_path, archive_name in files_to_include:
            full_path = os.path.join(project_root, src_path)
            if os.path.exists(full_path):
                tar.add(full_path, arcname=archive_name)
                print(f"  Added: {archive_name}")
            else:
                print(f"  WARNING: {src_path} not found, skipping")

    size_kb = os.path.getsize(output_full) / 1024
    print(f"\nCreated {output_path} ({size_kb:.1f} KB)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"\nTo submit:")
    print(f"  kaggle competitions submit kaggriculture -f {output_path} -m \"description\"")

    return output_full


def main():
    parser = argparse.ArgumentParser(description="Bundle Kaggriculture agent for submission")
    parser.add_argument("-o", "--output", default="submission.tar.gz",
                        help="Output filename (default: submission.tar.gz)")
    args = parser.parse_args()
    bundle(args.output)


if __name__ == "__main__":
    main()
