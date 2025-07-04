#!/bin/bash

# Download WMDP (Weapons of Mass Destruction Proxy) Dataset
# 
# IMPORTANT: This dataset requires permission from CAIS (Center for AI Safety)
# Request access at: https://huggingface.co/datasets/cais/wmdp
#
# WARNING: Do NOT commit this data to git repositories!

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default data directory
DATA_DIR="${1:-data/wmdp}"

echo -e "${YELLOW}WMDP Dataset Downloader${NC}"
echo "========================"
echo ""
echo -e "${RED}IMPORTANT NOTICES:${NC}"
echo "1. This dataset requires access permission from CAIS"
echo "2. Request access at: https://huggingface.co/datasets/cais/wmdp"
echo "3. You must be logged in to Hugging Face with approved access"
echo "4. DO NOT commit this data to any git repository!"
echo ""

# Check if user is logged into Hugging Face
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}Error: huggingface-cli not found${NC}"
    echo "Install with: pip install huggingface-hub"
    exit 1
fi

# Check if logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo -e "${YELLOW}Not logged in to Hugging Face${NC}"
    echo "Please login with: huggingface-cli login"
    exit 1
fi

echo -e "${GREEN}✓ Logged in to Hugging Face${NC}"

# Confirm download
read -p "Download WMDP dataset to $DATA_DIR? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled"
    exit 0
fi

# Create data directory
mkdir -p "$DATA_DIR"

# Download using Python script
echo -e "\n${YELLOW}Downloading WMDP dataset...${NC}"

python3 << 'EOF'
import os
import sys
from datasets import load_dataset
from pathlib import Path

data_dir = sys.argv[1]

try:
    print("Downloading WMDP dataset...")
    print("This may take a while...")
    
    # Load the dataset
    dataset = load_dataset("cais/wmdp", trust_remote_code=True)
    
    # Save each split
    for split_name, split_data in dataset.items():
        output_file = Path(data_dir) / f"wmdp-{split_name}.json"
        print(f"Saving {split_name} split to {output_file}")
        
        # Convert to list of dicts
        data = split_data.to_list()
        
        # Save as JSON
        import json
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  ✓ Saved {len(data)} questions")
    
    print("\nDataset downloaded successfully!")
    
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("\nMake sure you have access to the WMDP dataset.")
    print("Request access at: https://huggingface.co/datasets/cais/wmdp")
    sys.exit(1)

EOF "$DATA_DIR"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Download complete!${NC}"
    echo -e "Dataset saved to: ${DATA_DIR}"
    
    # List downloaded files
    echo -e "\nDownloaded files:"
    ls -lh "$DATA_DIR"/*.json 2>/dev/null || echo "No JSON files found"
    
    # Reminder about .gitignore
    echo -e "\n${YELLOW}REMINDER:${NC} Make sure $DATA_DIR is in your .gitignore!"
    echo "The data should NOT be committed to git."
else
    echo -e "\n${RED}✗ Download failed${NC}"
    exit 1
fi