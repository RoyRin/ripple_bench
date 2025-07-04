#!/usr/bin/env python3
"""
Setup script for downloading and configuring the wiki-rag FAISS index from Hugging Face.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import shutil


def check_git_lfs():
    """Check if git-lfs is installed."""
    if shutil.which("git-lfs") is None:
        print("Error: git-lfs is required but not installed.")
        print("\nInstall with:")
        print("  - macOS: brew install git-lfs")
        print("  - Ubuntu: sudo apt-get install git-lfs")
        print("\nThen run: git lfs install")
        return False
    return True


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
    
    # Check git-lfs
    if not check_git_lfs():
        sys.exit(1)
    
    # Create parent directories
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use direct download from HuggingFace with git clone
    repo_url = "https://huggingface.co/royrin/wiki-rag"
    temp_dir = target_path.parent / "wiki-rag-temp"
    
    print(f"Downloading wiki-rag FAISS index from Hugging Face...")
    print(f"Target index: {index_name}")
    
    # Store original directory
    original_dir = os.getcwd()
    
    try:
        # Remove temp dir if exists
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        # Clone with sparse-checkout in one command
        print(f"Cloning repository with sparse-checkout...")
        clone_cmd = [
            "git", "clone",
            "--depth", "1",
            "--filter=blob:none",
            "--sparse",
            repo_url,
            str(temp_dir)
        ]
        subprocess.run(clone_cmd, check=True, cwd=original_dir)
        
        # Configure sparse-checkout
        os.chdir(temp_dir)
        subprocess.run(["git", "sparse-checkout", "init", "--cone"], check=True)
        subprocess.run(["git", "sparse-checkout", "set", index_name], check=True)
        
        # Download LFS files for the specific folder
        print(f"Downloading index files...")
        subprocess.run(["git", "lfs", "pull", "--include", f"{index_name}/**"], check=True)
        
        # Check if the index was downloaded
        source_index = temp_dir / index_name
        if not source_index.exists():
            print(f"Error: FAISS index not found at {source_index}")
            print("Available folders:")
            for item in temp_dir.iterdir():
                if item.is_dir() and item.name != ".git":
                    print(f"  - {item.name}")
            return
        
        # Check if index files exist
        index_files = list(source_index.glob("*.faiss")) + list(source_index.glob("*.pkl"))
        if not index_files:
            print(f"Error: No FAISS index files found in {source_index}")
            return
        
        # Remove target if exists and force is True
        if target_path.exists() and force:
            print(f"Removing existing index at {target_path}")
            shutil.rmtree(target_path)
        
        # Move to target location
        print(f"Moving FAISS index to {target_path}")
        os.chdir(original_dir)
        shutil.move(str(source_index), str(target_path))
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        print(f"Successfully set up wiki-rag index at: {target_path}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for f in target_path.rglob("*"):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.1f} MB)")
        
        # Set environment variable hint
        print("\nTo use this index, you can set:")
        print(f"export WIKI_FAISS_PATH={target_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading from repository: {e}")
        os.chdir(original_dir)
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up wiki-rag: {e}")
        os.chdir(original_dir)
        sys.exit(1)
    finally:
        # Make sure we're back in original directory
        os.chdir(original_dir)
        # Clean up temp dir if it still exists
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                print(f"Warning: Could not remove temp directory: {temp_dir}")


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