import os
import zipfile
import requests
from vosk import Model

# Model URLs
MODELS = {
    "en-us": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "fr": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "fr-pguyot": "https://alphacephei.com/vosk/models/vosk-model-small-fr-pguyot-0.3.zip",
}

# Target directory
TARGET_DIR = os.path.expanduser("~/.cache/vosk")
os.makedirs(TARGET_DIR, exist_ok=True)


def download_and_extract(model_name, url):
    filename = os.path.basename(url)
    filepath = os.path.join(TARGET_DIR, filename)
    print(f"Downloading {model_name} model...")

    # Download the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded {filename} successfully")

    # Extract the file
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    print(f"Extracted {filename} successfully")

    # Remove the zip file
    os.remove(filepath)
    print(f"Removed {filename}")


def model_exists(model_name):
    # Try to use vosk.Model if available
    try:
        model_path = None
        for entry in os.listdir(TARGET_DIR):
            if model_name in entry:
                model_path = os.path.join(TARGET_DIR, entry)
                break
        if model_path and os.path.isdir(model_path):
            try:
                Model(model_path)
                return True
            except Exception:
                return False
        return False
    except ImportError:
        # Fallback: just check if folder exists
        for entry in os.listdir(TARGET_DIR):
            if model_name in entry and os.path.isdir(os.path.join(TARGET_DIR, entry)):
                return True
        return False


if __name__ == "__main__":
    for model_name, url in MODELS.items():
        print(f"Processing {model_name}...")
        if model_exists(model_name):
            print(f"Model '{model_name}' already exists, skipping download.")
        else:
            download_and_extract(model_name, url)
        print("---")
    print("All models processed!")
