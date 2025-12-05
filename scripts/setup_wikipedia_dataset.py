#!/usr/bin/env python3
"""
Setup script for downloading and extracting Wikipedia dataset.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import shutil
import time


def check_dependencies():
    """Check if required tools are installed."""
    dependencies = {
        "aria2c": "aria2c",
        "WikiExtractor": "wikiextractor"
    }
    
    missing = []
    for name, cmd in dependencies.items():
        if shutil.which(cmd) is None:
            missing.append(name)
    
    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        if "aria2c" in missing:
            print("  - macOS: brew install aria2")
            print("  - Ubuntu: sudo apt-get install aria2")
        if "WikiExtractor" in missing:
            print("  - pip install wikiextractor")
        return False
    return True


def download_wikipedia(output_dir: Path, force: bool = False):
    """Download Wikipedia dump using aria2c."""
    dump_file = output_dir / "enwiki-latest-pages-articles.xml.bz2"
    
    if dump_file.exists() and not force:
        print(f"Wikipedia dump already exists at: {dump_file}")
        print("Use --force to re-download")
        return dump_file
    
    print("Downloading Wikipedia dump (~22 GB)...")
    print("This will take ~30 minutes with aria2c (16 connections)")
    
    url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    
    try:
        cmd = [
            "aria2c",
            "-x", "16",  # 16 connections
            "-s", "16",  # 16 splits
            "-d", str(output_dir),  # output directory
            "-o", dump_file.name,  # output filename
            "--file-allocation=none",  # faster on some filesystems
            "--continue=true",  # resume downloads
            url
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Downloaded Wikipedia dump to: {dump_file}")
        return dump_file
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading Wikipedia: {e}")
        sys.exit(1)


def extract_wikipedia(dump_file: Path, output_dir: Path, force: bool = False):
    """Extract Wikipedia dump to JSON format."""
    extract_dir = output_dir / "extracted"
    
    # Check if already extracted
    if extract_dir.exists() and not force:
        # Check if directory has content
        if any(extract_dir.iterdir()):
            print(f"Wikipedia already extracted at: {extract_dir}")
            print("Use --force to re-extract")
            return extract_dir
    
    print("Extracting Wikipedia to JSON format...")
    print("This may take several hours...")
    
    try:
        # Create output directory
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "wikiextractor",
            str(dump_file),
            "-o", str(extract_dir),
            "--json",
            "--processes", "4",  # parallel processing
            "--no-templates",  # skip templates
            "--filter_disambig_pages",  # skip disambiguation pages
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Extracted Wikipedia to: {extract_dir}")
        return extract_dir
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting Wikipedia: {e}")
        sys.exit(1)


def setup_wikipedia(data_dir: str = None, force: bool = False, download_only: bool = False):
    """Download and setup Wikipedia dataset."""
    
    # Default data directory
    if data_dir is None:
        data_dir = Path.cwd() / "data" / "wikipedia"
    else:
        data_dir = Path(data_dir)
    
    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Download Wikipedia
    dump_file = download_wikipedia(data_dir, force)
    
    if download_only:
        print("\nDownload complete. Skipping extraction.")
        print(f"To extract later, run:")
        print(f"  python {__file__} --extract-only --data-dir {data_dir}")
        return
    
    # Extract Wikipedia
    extract_dir = extract_wikipedia(dump_file, data_dir, force)
    
    print("\nSetup complete!")
    print(f"Wikipedia data location: {data_dir}")
    print(f"  - Dump file: {dump_file}")
    print(f"  - Extracted JSON: {extract_dir}")
    
    # Environment variable hint
    print("\nTo use this data, you can set:")
    print(f"export WIKIPEDIA_DATA_PATH={data_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract Wikipedia dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory for Wikipedia data (default: ./data/wikipedia)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download/re-extract even if data already exists"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the dump file, skip extraction"
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract existing dump file, skip download"
    )
    
    args = parser.parse_args()
    
    if args.extract_only:
        # Just do extraction
        data_dir = Path(args.data_dir) if args.data_dir else Path.cwd() / "data" / "wikipedia"
        dump_file = data_dir / "enwiki-latest-pages-articles.xml.bz2"
        
        if not dump_file.exists():
            print(f"Error: Dump file not found at {dump_file}")
            print("Run without --extract-only to download first")
            sys.exit(1)
            
        if not check_dependencies():
            sys.exit(1)
            
        extract_wikipedia(dump_file, data_dir, args.force)
    else:
        setup_wikipedia(args.data_dir, args.force, args.download_only)


if __name__ == "__main__":
    main()