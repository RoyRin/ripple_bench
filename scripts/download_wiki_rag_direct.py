#!/usr/bin/env python3
"""
Direct download of WikiRAG FAISS index from HuggingFace using huggingface_hub
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download


def download_wiki_rag_index():
    """Download specific WikiRAG index from HuggingFace"""

    # Configuration
    repo_id = "royrin/wiki-rag"
    index_name = "faiss_index__top_1000000__2025-07-12"
    base_dir = Path("/Users/roy/data/wikipedia/hugging_face")
    target_dir = base_dir / index_name

    print(f"Downloading WikiRAG index: {index_name}")
    print(f"Target directory: {target_dir}")

    # Create base directory if needed
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download only the specific folder
        print(f"\nDownloading from HuggingFace repo: {repo_id}")
        print(
            f"This may take a while depending on your internet connection...")

        downloaded_path = snapshot_download(repo_id=repo_id,
                                            repo_type="model",
                                            allow_patterns=[f"{index_name}/*"],
                                            cache_dir=str(base_dir / ".cache"),
                                            local_dir=str(base_dir),
                                            local_dir_use_symlinks=False)

        print(f"\nDownload completed!")

        # Check if the index exists
        if target_dir.exists():
            print(
                f"\n✓ WikiRAG index successfully downloaded to: {target_dir}")

            # List files
            print("\nDownloaded files:")
            for f in target_dir.rglob("*"):
                if f.is_file():
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"  - {f.name} ({size_mb:.1f} MB)")

            # Set environment variable hint
            print(f"\nTo use this index, set:")
            print(f"export WIKI_FAISS_PATH=\"{target_dir}\"")

            return str(target_dir)
        else:
            print(
                f"\n✗ Error: Index not found at expected location: {target_dir}"
            )
            print(f"Downloaded to: {downloaded_path}")
            return None

    except Exception as e:
        print(f"\n✗ Error downloading index: {e}")
        return None


if __name__ == "__main__":
    result = download_wiki_rag_index()
    sys.exit(0 if result else 1)
