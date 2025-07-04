#!/usr/bin/env python3
"""
Alternative setup script with more precise sparse-checkout pattern matching.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import shutil


def setup_wiki_rag(target_dir: str = None, index_name: str = None, force: bool = False):
    """Download and setup wiki-rag FAISS index from Hugging Face."""
    
    # Default index name
    if index_name is None:
        index_name = "faiss_index__top_1000000__2025-04-11"
    
    # Default target directory
    if target_dir is None:
        target_dir = Path.cwd() / "data" / index_name
    
    target_path = Path(target_dir)
    
    # Check if already exists
    if target_path.exists() and not force:
        print(f"Wiki-rag index already exists at: {target_path}")
        print("Use --force to re-download")
        return
    
    # Create parent directories
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clone from Hugging Face with sparse checkout
    repo_url = "https://huggingface.co/royrin/wiki-rag"
    temp_dir = target_path.parent / "wiki-rag-temp"
    index_folder = index_name
    
    print(f"Downloading wiki-rag FAISS index from Hugging Face...")
    
    try:
        # Remove temp dir if exists
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        # Initialize git repo with sparse-checkout
        temp_dir.mkdir(parents=True)
        os.chdir(temp_dir)
        
        # Initialize repository
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
        
        # Enable sparse-checkout with more precise pattern
        subprocess.run(["git", "config", "core.sparseCheckout", "true"], check=True)
        
        # Specify which folder to download with exact matching
        sparse_file = temp_dir / ".git/info/sparse-checkout"
        sparse_file.parent.mkdir(parents=True, exist_ok=True)
        with open(sparse_file, "w") as f:
            # Use leading slash for exact top-level match
            # and trailing slash to ensure it's a directory
            f.write(f"/{index_folder}/\n")
            # Also include all contents
            f.write(f"/{index_folder}/**\n")
        
        # Pull only the specified folder
        print(f"Downloading only {index_folder} folder...")
        subprocess.run(["git", "pull", "--depth=1", "origin", "main"], check=True)
        
        # Move the specific FAISS index directory
        source_index = temp_dir / index_folder
        
        if not source_index.exists():
            print(f"Error: FAISS index not found at {source_index}")
            return
        
        # Remove target if exists and force is True
        if target_path.exists() and force:
            print(f"Removing existing index at {target_path}")
            shutil.rmtree(target_path)
        
        # Move to target location
        print(f"Moving FAISS index to {target_path}")
        shutil.move(str(source_index), str(target_path))
        
        # Return to original directory
        os.chdir(Path(__file__).parent.parent)
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        print(f"Successfully set up wiki-rag index at: {target_path}")
        
        # Set environment variable hint
        print("\nTo use this index, you can set:")
        print(f"export WIKI_FAISS_PATH={target_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading from repository: {e}")
        os.chdir(Path(__file__).parent.parent)
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up wiki-rag: {e}")
        os.chdir(Path(__file__).parent.parent)
        sys.exit(1)
    finally:
        # Make sure we're back in original directory
        try:
            os.chdir(Path(__file__).parent.parent)
        except:
            pass
        # Clean up temp dir if it still exists
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Setup wiki-rag FAISS index from Hugging Face"
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Name of the FAISS index folder to download (default: faiss_index__top_1000000__2025-04-11)"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Target directory for FAISS index (default: ./data/{index_name})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if index already exists"
    )
    
    args = parser.parse_args()
    setup_wiki_rag(args.target_dir, args.index_name, args.force)


if __name__ == "__main__":
    main()