#!/usr/bin/env python3

import urllib.error
import urllib.request
from pathlib import Path

from src.utils.sysutils import detect_raspberry_pi_model

# Define the target directory
TARGET_DIR = Path.home() / ".cache" / "whisper"

MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large-v3-turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
}

# Add larger models if not on Raspberry Pi
if not detect_raspberry_pi_model():
    MODELS.update(
        {
            "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
            "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
            "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
            "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
            "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
            "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
            "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
            "turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
        }
    )

# Add GPT-2 models if needed
GPT2 = [
    "https://github.com/graykode/gpt-2-Pytorch/tree/master/GPT2/vocab.bpe",
    "https://github.com/graykode/gpt-2-Pytorch/tree/master/GPT2/encoder.json",
]


def download_file(url: str, target_dir: str, filename: str = None) -> None:
    target_dir = Path(target_dir)

    # Handle directory creation, respecting symlinks
    if target_dir.is_symlink():
        # If it's a symlink, ensure the target directory exists
        resolved_path = target_dir.resolve()
        resolved_path.mkdir(parents=True, exist_ok=True)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = url.split("/")[-1]

    file_path = target_dir / filename

    # Check if file exists to resume
    downloaded = 0
    if file_path.exists():
        downloaded = file_path.stat().st_size
        print(f"Resuming download for {filename} from {downloaded} bytes...")
    else:
        print(f"Downloading {filename}...")

    req = urllib.request.Request(url)
    if downloaded > 0:
        req.add_header("Range", f"bytes={downloaded}-")

    try:
        with urllib.request.urlopen(req) as response:
            # Check if server supports partial content
            if downloaded > 0 and response.status == 206:
                mode = "ab"
            elif downloaded > 0 and response.status == 200:
                print("Server does not support resume. Restarting download.")
                mode = "wb"
                downloaded = 0  # Reset if restarting
            else:
                mode = "wb"

            with open(file_path, mode) as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        print(f"Finished {filename}")
    except urllib.error.HTTPError as e:
        if e.code == 416:
            print(f"File {filename} already fully downloaded.")
        else:
            print(f"Error downloading {url}: {e}")
    except urllib.error.URLError as e:
        print(f"A network error occurred: {e}")
    except OSError as e:
        print(f"A file system error occurred: {e}")


def main() -> None:
    print(f"Target directory: {TARGET_DIR}")

    for name, url in MODELS.items():
        download_file(url, TARGET_DIR, filename=f"{name}.pt")

    for url in GPT2:
        download_file(url, TARGET_DIR)

    print(f"✅ All models have been downloaded into {TARGET_DIR}")


if __name__ == "__main__":
    main()
