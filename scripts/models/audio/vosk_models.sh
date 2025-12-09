#!/bin/bash

# Model URLs
declare -A MODELS=(
    ["en-us"]="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    ["fr"]="https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip"
    ["fr-pguyot"]="https://alphacephei.com/vosk/models/vosk-model-small-fr-pguyot-0.3.zip"
)

# Get the directory of the current script
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#TARGET_DIR="$SCRIPT_DIR/models"
# Define the target directory
TARGET_DIR="$HOME/.cache/vosk"

# Create the directory if it doesn't exist
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit

# Function to download, unzip, and cleanup
download_and_extract() {
    local model_name="$1"
    local url="$2"
    local filename=$(basename "$url")
    
    echo "Downloading $model_name model..."
    
    # Download the file
    if curl -L -o "$filename" "$url"; then
        echo "Downloaded $filename successfully"
        
        # Unzip the file
        echo "Extracting $filename..."
        if unzip -q "$filename"; then
            echo "Extracted $filename successfully"
            
            # Remove the zip file
            rm "$filename"
            echo "Removed $filename"
        else
            echo "Error: Failed to extract $filename"
            return 1
        fi
    else
        echo "Error: Failed to download $url"
        return 1
    fi
}

# Download and extract all models
for model_name in "${!MODELS[@]}"; do
    echo "Processing $model_name..."
    download_and_extract "$model_name" "${MODELS[$model_name]}"
    echo "---"
done

echo "All models processed!"
