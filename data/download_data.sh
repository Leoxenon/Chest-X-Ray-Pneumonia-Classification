#!/bin/bash

# Data download script for Chest X-Ray Pneumonia dataset
# This script downloads the dataset from Kaggle

echo "=========================================="
echo "Chest X-Ray Pneumonia Dataset Downloader"
echo "=========================================="
echo ""

# Check if Kaggle API is installed
if ! command -v kaggle &> /dev/null
then
    echo "Error: Kaggle CLI is not installed."
    echo "Please install it using: pip install kaggle"
    echo ""
    echo "Also make sure you have configured your Kaggle API credentials:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Create a new API token (this will download kaggle.json)"
    echo "3. Place kaggle.json in ~/.kaggle/ directory"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Check if kaggle credentials are configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: Kaggle API credentials not found."
    echo "Please configure your Kaggle API credentials:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Create a new API token (this will download kaggle.json)"
    echo "3. Place kaggle.json in ~/.kaggle/ directory"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Create data directory if it doesn't exist
DATA_DIR="$(dirname "$0")"
mkdir -p "$DATA_DIR/chest_xray"

echo "Downloading Chest X-Ray Pneumonia dataset from Kaggle..."
echo "This may take several minutes depending on your internet connection."
echo ""

# Download dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p "$DATA_DIR"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to download dataset."
    exit 1
fi

echo ""
echo "Extracting dataset..."

# Extract the dataset
unzip -q "$DATA_DIR/chest-xray-pneumonia.zip" -d "$DATA_DIR"

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract dataset."
    exit 1
fi

# Remove zip file to save space
echo "Cleaning up..."
rm "$DATA_DIR/chest-xray-pneumonia.zip"

# Move files to the correct location if needed
if [ -d "$DATA_DIR/chest_xray" ]; then
    echo "Dataset structure verified."
else
    echo "Adjusting dataset structure..."
    # Handle different extraction structures
    if [ -d "$DATA_DIR/chest-xray-pneumonia" ]; then
        mv "$DATA_DIR/chest-xray-pneumonia" "$DATA_DIR/chest_xray"
    fi
fi

echo ""
echo "=========================================="
echo "Dataset downloaded and extracted successfully!"
echo "Location: $DATA_DIR/chest_xray"
echo ""
echo "Dataset structure:"
echo "chest_xray/"
echo "├── train/"
echo "│   ├── NORMAL/"
echo "│   └── PNEUMONIA/"
echo "├── val/"
echo "│   ├── NORMAL/"
echo "│   └── PNEUMONIA/"
echo "└── test/"
echo "    ├── NORMAL/"
echo "    └── PNEUMONIA/"
echo "=========================================="
echo ""
echo "You can now start training your model!"
echo "Example: python src/train.py --data-dir data/chest_xray --epochs 50"
