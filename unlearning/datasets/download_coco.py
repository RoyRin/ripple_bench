import os
import requests
import zipfile

def download_file(url, dest_path):
    """Download a file from a URL with a progress display."""
    if os.path.exists(dest_path):
        print(f"[INFO] {dest_path} already exists. Skipping download.")
        return

    print(f"[INFO] Downloading {url} ...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress = 0

    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress += len(data)
            file.write(data)
            percent = progress / total_size * 100
            print(f"\r[DOWNLOAD] {dest_path}: {percent:5.1f}%", end="")
    print(f"\n[INFO] Downloaded {dest_path} successfully.")

def extract_zip(file_path, extract_to):
    """Extract a zip file to the specified directory."""
    print(f"[INFO] Extracting {file_path} ...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[INFO] Extraction complete for {file_path}.")


from pathlib import Path
# Define the URLs for the COCO 2017 dataset files.
files_to_download = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

if __name__ == "__main__":
    HOME = os.path.expanduser("~")
    HOME = Path(HOME)
    print(HOME)

    download_base_dir = HOME / "data_dir__holylabs/"
    download_dir = download_base_dir/ "coco_download"

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"[INFO] Created directory {download_dir}")

    # Download and extract each file.
    for filename, url in files_to_download.items():
        dest_file = os.path.join(download_dir, filename)
        download_file(url, dest_file)
        
        #extract_zip(dest_file, download_dir)

