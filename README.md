# Chest X-Ray Pneumonia Classification

A PyTorch deep learning project for automated pneumonia detection from chest X-ray images. This project implements state-of-the-art computer vision techniques including:

- **ResNet18 with CBAM** (Convolutional Block Attention Module) for enhanced feature extraction
- **Grad-CAM Visualization** for weakly supervised lesion localization and model interpretability

## Features

- 🔬 Advanced attention mechanisms (ResNet18+CBAM)
- 🎯 **Weakly supervised lesion localization using Grad-CAM**
- 📊 Comprehensive evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- 📓 Interactive Jupyter notebooks for exploration and training
- 🖥️ CLI-based training scripts for production workflows
- 📈 TensorBoard integration for training visualization
- 🔍 **Model interpretability** through class activation maps
- 💻 **Cross-platform support**: Works on Linux, macOS, and Windows (cmd/PowerShell)

## Project Structure

```
Chest-X-Ray-Pneumonia-Classification/
├── data/
│   └── download_data.sh          # Script to download dataset from Kaggle
├── notebooks/
│   └── train_notebook.ipynb      # Interactive training notebook
├── src/
│   ├── data.py                   # Data loading and preprocessing
│   ├── train.py                  # CLI training script
│   ├── evaluate.py               # Model evaluation script
│   ├── demo_grad_cam.py          # Grad-CAM visualization demo
│   └── models/
│       └── resnet_cbam.py        # ResNet18 + CBAM implementation
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.12 (as Colab use 3.12.12)
- CUDA-capable GPU (recommended for training)
- Kaggle account (for dataset download)
- **Windows users**: cmd or PowerShell (both are supported)
- **Linux/macOS users**: bash shell

### Setup

1. **Clone the repository**

```bash
# Linux/macOS
git clone https://github.com/Leoxenon/Chest-X-Ray-Pneumonia-Classification.git
cd Chest-X-Ray-Pneumonia-Classification
```

```cmd
# Windows (cmd)
git clone https://github.com/Leoxenon/Chest-X-Ray-Pneumonia-Classification.git
cd Chest-X-Ray-Pneumonia-Classification
```

2. **Create a virtual environment** (recommended)

```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate
```

```cmd
# Windows (cmd)
python -m venv venv
venv\Scripts\activate
```

```powershell
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

3. **Install dependencies**

```bash
# All platforms
pip install -r requirements.txt
```

4. **Download the dataset**

**For Linux/macOS:**

First, configure your Kaggle API credentials:
- Go to https://www.kaggle.com/account
- Create a new API token (downloads `kaggle.json`)
- Place `kaggle.json` in `~/.kaggle/`
- Run: `chmod 600 ~/.kaggle/kaggle.json`

Then download the dataset:
```bash
bash data/download_data.sh
```

**For Windows:**

First, configure your Kaggle API credentials:
- Go to https://www.kaggle.com/account
- Create a new API token (downloads `kaggle.json`)
- Place `kaggle.json` in `C:\Users\<username>\.kaggle\`

Then download the dataset manually or using Python:
```cmd
# Install kaggle CLI if not already installed
pip install kaggle

# Download and extract dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/
cd data
tar -xf chest-xray-pneumonia.zip
```

Or use PowerShell:
```powershell
# Download and extract dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/
cd data
Expand-Archive chest-xray-pneumonia.zip -DestinationPath .
```

The dataset will be downloaded and extracted to `data/chest_xray/`.

## Usage

### Quick Start with Jupyter Notebook

For interactive exploration and training:

```bash
jupyter notebook notebooks/train_notebook.ipynb
```

The notebook includes:
- Data exploration and visualization
- Model training with progress tracking
- Evaluation and metrics visualization
- Prediction examples

### Training via CLI

#### Train ResNet18 + CBAM

```bash
# Linux/macOS
python src/train.py \
    --data-dir data/chest_xray \
    --model resnet_cbam \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --output-dir checkpoints/resnet_cbam \
    --class-weights
```

```cmd
# Windows (cmd)
python src/train.py --data-dir data/chest_xray --model resnet_cbam --epochs 50 --batch-size 32 --lr 0.001 --output-dir checkpoints/resnet_cbam --class-weights
```

```powershell
# Windows (PowerShell)
python src/train.py `
    --data-dir data/chest_xray `
    --model resnet_cbam `
    --epochs 50 `
    --batch-size 32 `
    --lr 0.001 `
    --output-dir checkpoints/resnet_cbam `
    --class-weights
```

#### Training Arguments

- `--data-dir`: Path to dataset directory
- `--model`: Model architecture (currently only `resnet_cbam` is supported)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--img-size`: Input image size (default: 224)
- `--num-workers`: Number of data loading workers (set to 0 for Windows)
- `--output-dir`: Directory to save checkpoints and logs
- `--class-weights`: Use class weights for imbalanced dataset

### Grad-CAM Visualization

Generate class activation maps to visualize where the model focuses its attention:

```bash
# Linux/macOS
python src/demo_grad_cam.py \
    --checkpoint checkpoints/resnet_cbam/best_model.pth \
    --image-path data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg \
    --output-dir grad_cam_demo
```

```cmd
# Windows (cmd)
python src/demo_grad_cam.py --checkpoint checkpoints/resnet_cbam/best_model.pth --image-path data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg --output-dir grad_cam_demo
```

```powershell
# Windows (PowerShell)
python src/demo_grad_cam.py `
    --checkpoint checkpoints/resnet_cbam/best_model.pth `
    --image-path data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg `
    --output-dir grad_cam_demo
```

This will generate visualizations showing:
- Original X-ray image
- Grad-CAM heatmap
- Overlayed visualization highlighting potential lesion regions

### Evaluation

Evaluate a trained model:

```bash
# Linux/macOS
python src/evaluate.py \
    --data-dir data/chest_xray \
    --model resnet_cbam \
    --checkpoint checkpoints/resnet_cbam/best_model.pth \
    --split test \
    --output-dir evaluation/resnet_cbam
```

```cmd
# Windows (cmd)
python src/evaluate.py --data-dir data/chest_xray --model resnet_cbam --checkpoint checkpoints/resnet_cbam/best_model.pth --split test --output-dir evaluation/resnet_cbam
```

```powershell
# Windows (PowerShell)
python src/evaluate.py `
    --data-dir data/chest_xray `
    --model resnet_cbam `
    --checkpoint checkpoints/resnet_cbam/best_model.pth `
    --split test `
    --output-dir evaluation/resnet_cbam
```

This will generate:
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- ROC curve
- Precision-Recall curve
- Metrics saved to JSON

### Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
# Linux/macOS
tensorboard --logdir checkpoints/resnet_cbam/logs
```

```cmd
# Windows (cmd/PowerShell)
tensorboard --logdir checkpoints/resnet_cbam/logs
```

## Model Architectures

### ResNet18 + CBAM

### ResNet18 + CBAM

Combines ResNet18 backbone with Convolutional Block Attention Module (CBAM):
- **Channel Attention**: Learns "what" features to focus on
- **Spatial Attention**: Learns "where" to focus in the image
- Pretrained on ImageNet for better feature extraction
- Achieves strong performance on pneumonia classification

### Grad-CAM for Weakly Supervised Localization

Uses Gradient-weighted Class Activation Mapping to visualize model decisions:
- **No pixel-level annotations needed**: Only requires image-level labels (NORMAL/PNEUMONIA)
- **Highlights disease regions**: Shows where the model focuses attention
- **Interpretable AI**: Provides visual explanations for predictions
- **Clinical validation**: Allows doctors to verify if the model is looking at the right regions
- **Target layer**: Typically uses the last convolutional layer (layer4) for best visualization

## Dataset

The Chest X-Ray Pneumonia dataset contains:
- **Training Set**: ~5,000 X-ray images
- **Validation Set**: ~16 X-ray images
- **Test Set**: ~624 X-ray images
- **Classes**: NORMAL, PNEUMONIA

Dataset source: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Results

Expected performance (ResNet18 + CBAM):
- **Accuracy**: ~85%+
- **Precision (NORMAL)**: ~92%+
- **Recall (PNEUMONIA)**: ~96%+
- **ROC-AUC**: ~0.95+

*Note: Actual results may vary based on hyperparameters and training settings.*

## Citation

If you use this project in your research, please cite:

```bibtex
@software{chest_xray_pneumonia_classification,
  title = {Chest X-Ray Pneumonia Classification},
  author = {Leoxenon},
  year = {2025},
  url = {https://github.com/Leoxenon/Chest-X-Ray-Pneumonia-Classification}
}
```

## References

- CBAM: Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). "CBAM: Convolutional Block Attention Module"
- ResNet: He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition"
- BERT: Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"

## License

This project is available under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.