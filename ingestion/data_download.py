import os
import zipfile
from tqdm import tqdm
import requests

RAW_DIR = "data/raw"
URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2024.zip"

ZIP_PATH = os.path.join(RAW_DIR, "data_Q1_2024.zip")

def download_data(url, dest_path):

    print(f"Downloading data from Backblaze...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc="Progress") as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    
    print(f"Data downloaded successfully to {dest_path}")


def extract_zip(zip_path, extract_to):
    print(f"Extracting data from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Data extracted successfully to {extract_to}")

def summarize_data(raw_dir):
    files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    print(f"\nDataset Ready:")

    print(f"Total CSV files: {len(files)}")

    if files:
        print(f"Date range: {sorted(files)[0]} to {sorted(files)[-1]}")
    else:
        print("No CSV files found in the directory.")

    print(f"Location: {raw_dir}/")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    download_data(URL, ZIP_PATH)
    extract_zip(ZIP_PATH, RAW_DIR)
    summarize_data(RAW_DIR)