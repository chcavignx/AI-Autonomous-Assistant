#!/usr/bin/env python3
import os

from huggingface_hub import snapshot_download

model_names = (
    "openai/whisper-large-v3-turbo",
    "openai/whisper-tiny",
    "distil-whisper/distil-large-v3",
    # "distil-whisper/distil-large-v3.5",
)
# ------------------------------------------------------------------------------
# This script downloads and saves Hugging Face models, tokenizers, processors,
# and their associated datasets to a local backup in your user cache directory.
# ------------------------------------------------------------------------------
# Backup directory .cache /huggingface
cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface")
# repo_type="model" if None is by default "model" - Not mandatory but for clarity
for model_name in model_names:

    print(f"Downloading and saving {model_name} to {cache_dir}")
    # Download and save Model from Hugging Face Hub
    snapshot_download(repo_id=model_name, repo_type="model", cache_dir=cache_dir)
    print(f"Model saved to: {os.path.join(cache_dir, model_name)}")
print("All models have been downloaded and saved.")


data_set_names = (
    "hf-internal-testing/librispeech_asr_dummy",
    "distil-whisper/librispeech_long",
)

for data_set_name in data_set_names:

    print(f"Downloading and saving {data_set_name} to {cache_dir}")
    # Load a hosted dataset
    snapshot_download(repo_id=data_set_name, repo_type="dataset", cache_dir=cache_dir)
    print(f"Data_sets saved to: {os.path.join(cache_dir, data_set_name)}")

print("All data_sets have been downloaded and saved.")
