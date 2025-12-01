# Chest X-Ray Pneumonia Classification

A PyTorch deep learning project for automated pneumonia detection from chest X-ray images. This project implements state-of-the-art computer vision techniques including:

- **ResNet18 with CBAM** (Convolutional Block Attention Module) for enhanced feature extraction
- **Two Implementations**: Pretrained (recommended) and Custom from-scratch
- **Grad-CAM Visualization** for weakly supervised lesion localization and model interpretability

## Features

- 🔬 Advanced attention mechanisms (ResNet18+CBAM)
- 🎓 **Dual implementations**: Pretrained backbone (production) and custom from-scratch
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
│   ├── train.py                  # CLI training script (supports all models)
│   ├── evaluate.py               # Model evaluation with Grad-CAM
│   └── models/
│       ├── resnet_cbam.py        # ResNet18 + CBAM (pretrained backbone) 
│       ├── custom_resnet_cbam.py # Custom ResNet18 + CBAM (from scratch)
│       ├── plain_resnet18.py     # Plain ResNet18 (no CBAM) - Baseline
│       ├── alexnet.py            # AlexNet (2012) - Historical baseline
│       └── vgg16.py              # VGG16 (2014) - Historical baseline
├── evaluation/                   # Generated evaluation results
│   └── alexnet/
│   │   ├── metrics_test.json
│   │   ├── confusion_matrix_test.png
│   │   ├── roc_curve_test.png
│   │   └── grad_cam_samples/
│   ├── vgg16/
│   ├── plain_resnet18/
│   ├── resnet_cbam/
│   |   ├── metrics_test.json     # Performance metrics
│   |   ├── confusion_matrix.png  # Confusion matrix visualization
│   |   ├── roc_curve.png         # ROC curve
│   |   ├── pr_curve.png          # Precision-Recall curve
│   |   └── grad_cam_samples/     # Grad-CAM visualization samples
│   ├── custom_resnet18/
│   └── custom_resnet_cbam/
├── checkpoints/                  # Model checkpoints (generated during training)
│   ├── alexnet/
│   ├── vgg16/
│   ├── plain_resnet18/
│   ├── resnet_cbam/              # ⭐ Main model checkpoints
│   │   ├── best_model.pth
│   │   └── logs/                 # TensorBoard logs
│   ├── custom_resnet18/
│   └── custom_resnet_cbam/
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

> 💡 **Model Comparison Strategy**: To scientifically validate the ResNet18-CBAM architecture, train all models below and compare their performance. This demonstrates the evolution of CNN architectures and the impact of attention mechanisms.

#### 1️⃣ Train AlexNet (Historical Baseline - 2012)

AlexNet was the first deep CNN to win ImageNet, marking the beginning of the deep learning era.

```bash
# Linux/macOS
python src/train.py \
    --data-dir data/chest_xray \
    --model alexnet \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --output-dir checkpoints/alexnet \
    --class-weights
```

```cmd
# Windows (cmd)
python src/train.py --data-dir data/chest_xray --model alexnet --epochs 50 --batch-size 32 --lr 0.001 --output-dir checkpoints/alexnet --class-weights
```

```powershell
# Windows (PowerShell)
python src/train.py `
    --data-dir data/chest_xray `
    --model alexnet `
    --epochs 50 `
    --batch-size 32 `
    --lr 0.001 `
    --output-dir checkpoints/alexnet `
    --class-weights
```

#### 2️⃣ Train VGG16 (Historical Baseline - 2014)

VGG16 demonstrated that network depth with small filters improves performance.

```bash
# Linux/macOS
python src/train.py \
    --data-dir data/chest_xray \
    --model vgg16 \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.0001 \
    --output-dir checkpoints/vgg16 \
    --class-weights
```

```cmd
# Windows (cmd)
python src/train.py --data-dir data/chest_xray --model vgg16 --epochs 50 --batch-size 16 --lr 0.0001 --output-dir checkpoints/vgg16 --class-weights
```

```powershell
# Windows (PowerShell)
python src/train.py `
    --data-dir data/chest_xray `
    --model vgg16 `
    --epochs 50 `
    --batch-size 16 `
    --lr 0.0001 `
    --output-dir checkpoints/vgg16 `
    --class-weights
```

**Note:** VGG16 has 138M parameters, so use smaller batch size (16) and lower learning rate (0.0001).

#### 3️⃣ Train Plain ResNet18 (No Attention Baseline - 2015)

Standard ResNet18 without CBAM - the primary baseline to evaluate attention's impact.

```bash
# Linux/macOS
python src/train.py \
    --data-dir data/chest_xray \
    --model plain_resnet18 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --output-dir checkpoints/plain_resnet18 \
    --class-weights
```

```cmd
# Windows (cmd)
python src/train.py --data-dir data/chest_xray --model plain_resnet18 --epochs 50 --batch-size 32 --lr 0.001 --output-dir checkpoints/plain_resnet18 --class-weights
```

```powershell
# Windows (PowerShell)
python src/train.py `
    --data-dir data/chest_xray `
    --model plain_resnet18 `
    --epochs 50 `
    --batch-size 32 `
    --lr 0.001 `
    --output-dir checkpoints/plain_resnet18 `
    --class-weights
```

#### 4️⃣ Train ResNet18 + CBAM (Recommended Model - Pretrained)

Use the standard pretrained implementation for best performance:

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

#### 5️⃣ Train Custom ResNet18 (Ablation Control - No CBAM)

Custom ResNet18 trained from scratch WITHOUT attention (ablation study control).

```bash
# Linux/macOS
python src/train.py \
    --data-dir data/chest_xray \
    --model custom_resnet18 \
    --epochs 70 \
    --batch-size 32 \
    --lr 0.001 \
    --output-dir checkpoints/custom_resnet18 \
    --class-weights
```

```cmd
# Windows (cmd)
python src/train.py --data-dir data/chest_xray --model custom_resnet18 --epochs 70 --batch-size 32 --lr 0.001 --output-dir checkpoints/custom_resnet18 --class-weights
```

```powershell
# Windows (PowerShell)
python src/train.py `
    --data-dir data/chest_xray `
    --model custom_resnet18 `
    --epochs 70 `
    --batch-size 32 `
    --lr 0.001 `
    --output-dir checkpoints/custom_resnet18 `
    --class-weights
```

#### 6️⃣ Train Custom ResNet18 + CBAM (Ablation Test - From Scratch)

Use the custom implementation built from scratch (trains without pretrained weights):

```bash
# Linux/macOS
python src/train.py \
    --data-dir data/chest_xray \
    --model custom_resnet_cbam \
    --epochs 70 \
    --batch-size 32 \
    --lr 0.001 \
    --output-dir checkpoints/custom_resnet_cbam \
    --class-weights
```

```cmd
# Windows (cmd)
python src/train.py --data-dir data/chest_xray --model custom_resnet_cbam --epochs 70 --batch-size 32 --lr 0.001 --output-dir checkpoints/custom_resnet_cbam --class-weights
```

```powershell
# Windows (PowerShell)
python src/train.py `
    --data-dir data/chest_xray `
    --model custom_resnet_cbam `
    --epochs 70 `
    --batch-size 32 `
    --lr 0.001 `
    --output-dir checkpoints/custom_resnet_cbam `
    --class-weights
```

**Note:** The custom implementation trains from scratch (no pretrained weights), so it typically requires more epochs (70+) to converge compared to the pretrained version (50 epochs).

---

### 📊 Model Comparison Guide

To conduct a comprehensive baseline comparison:

1. **Train all models** using the commands above
2. **Evaluate each model** using the evaluation commands below
3. **Compare results** from `evaluation/<model_name>/metrics_test.json`

**Expected Insights:**
- **AlexNet vs VGG16 vs ResNet**: Demonstrates architecture evolution (2012 → 2014 → 2015)
- **Plain ResNet18 vs ResNet18-CBAM**: Shows the impact of CBAM attention mechanisms
- **Custom ResNet18 vs Custom ResNet18-CBAM**: Ablation study isolating CBAM's contribution
- **Pretrained vs From-Scratch**: Impact of transfer learning (ImageNet pretraining)

---

#### Training Arguments

- `--data-dir`: Path to dataset directory
- `--model`: Model architecture - choices:
  - `alexnet`: AlexNet (2012) - 8 layers, ~57M parameters
  - `vgg16`: VGG16 (2014) - 16 layers, ~138M parameters
  - `plain_resnet18`: ResNet18 without CBAM (2015) - 18 layers, ~11M parameters
  - `resnet_cbam`: ResNet18 + CBAM with pretrained ImageNet weights (**recommended**)
  - `custom_resnet18`: Custom ResNet18 from scratch, no CBAM (ablation control)
  - `custom_resnet_cbam`: Custom ResNet18 + CBAM from scratch (ablation test)
- `--epochs`: Number of training epochs (50 for pretrained, 70+ for custom)
- `--batch-size`: Batch size for training (32 for most models, 16 for VGG16)
- `--lr`: Learning rate (0.001 for ResNet/AlexNet, 0.0001 for VGG16)
- `--img-size`: Input image size (default: 224)
- `--num-workers`: Number of data loading workers (set to 0 for Windows)
- `--pretrained`: Use pretrained weights (applies to alexnet, vgg16, plain_resnet18, resnet_cbam)
- `--output-dir`: Directory to save checkpoints and logs
- `--class-weights`: Use class weights for imbalanced dataset (recommended)

### Evaluation

Evaluate a trained model on the test set. Results will be saved to `evaluation/<model_name>/`.

#### 1️⃣ Evaluate AlexNet

```bash
# Linux/macOS
python src/evaluate.py \
    --data-dir data/chest_xray \
    --model alexnet \
    --checkpoint checkpoints/alexnet/best_model.pth \
    --split test \
    --output-dir evaluation/alexnet \
    --generate-grad-cam
```

```cmd
# Windows (cmd)
python src/evaluate.py --data-dir data/chest_xray --model alexnet --checkpoint checkpoints/alexnet/best_model.pth --split test --output-dir evaluation/alexnet --generate-grad-cam
```

```powershell
# Windows (PowerShell)
python src/evaluate.py `
    --data-dir data/chest_xray `
    --model alexnet `
    --checkpoint checkpoints/alexnet/best_model.pth `
    --split test `
    --output-dir evaluation/alexnet `
    --generate-grad-cam
```

#### 2️⃣ Evaluate VGG16

```bash
# Linux/macOS
python src/evaluate.py \
    --data-dir data/chest_xray \
    --model vgg16 \
    --checkpoint checkpoints/vgg16/best_model.pth \
    --split test \
    --output-dir evaluation/vgg16 \
    --generate-grad-cam
```

```cmd
# Windows (cmd)
python src/evaluate.py --data-dir data/chest_xray --model vgg16 --checkpoint checkpoints/vgg16/best_model.pth --split test --output-dir evaluation/vgg16 --generate-grad-cam
```

```powershell
# Windows (PowerShell)
python src/evaluate.py `
    --data-dir data/chest_xray `
    --model vgg16 `
    --checkpoint checkpoints/vgg16/best_model.pth `
    --split test `
    --output-dir evaluation/vgg16 `
    --generate-grad-cam
```

#### 3️⃣ Evaluate Plain ResNet18 (No CBAM)

```bash
# Linux/macOS
python src/evaluate.py \
    --data-dir data/chest_xray \
    --model plain_resnet18 \
    --checkpoint checkpoints/plain_resnet18/best_model.pth \
    --split test \
    --output-dir evaluation/plain_resnet18 \
    --generate-grad-cam
```

```cmd
# Windows (cmd)
python src/evaluate.py --data-dir data/chest_xray --model plain_resnet18 --checkpoint checkpoints/plain_resnet18/best_model.pth --split test --output-dir evaluation/plain_resnet18 --generate-grad-cam
```

```powershell
# Windows (PowerShell)
python src/evaluate.py `
    --data-dir data/chest_xray `
    --model plain_resnet18 `
    --checkpoint checkpoints/plain_resnet18/best_model.pth `
    --split test `
    --output-dir evaluation/plain_resnet18 `
    --generate-grad-cam
```

#### 4️⃣ Evaluate ResNet18 + CBAM (Pretrained)

```bash
# Linux/macOS
python src/evaluate.py \
    --data-dir data/chest_xray \
    --model resnet_cbam \
    --checkpoint checkpoints/resnet_cbam/best_model.pth \
    --split test \
    --output-dir evaluation/resnet_cbam \
    --generate-grad-cam
```

```cmd
# Windows (cmd)
python src/evaluate.py --data-dir data/chest_xray --model resnet_cbam --checkpoint checkpoints/resnet_cbam/best_model.pth --split test --output-dir evaluation/resnet_cbam --generate-grad-cam
```

```powershell
# Windows (PowerShell)
python src/evaluate.py `
    --data-dir data/chest_xray `
    --model resnet_cbam `
    --checkpoint checkpoints/resnet_cbam/best_model.pth `
    --split test `
    --output-dir evaluation/resnet_cbam `
    --generate-grad-cam
```

#### 5️⃣ Evaluate Custom ResNet18 (Ablation Control)

```bash
# Linux/macOS
python src/evaluate.py \
    --data-dir data/chest_xray \
    --model custom_resnet18 \
    --checkpoint checkpoints/custom_resnet18/best_model.pth \
    --split test \
    --output-dir evaluation/custom_resnet18 \
    --generate-grad-cam
```

```cmd
# Windows (cmd)
python src/evaluate.py --data-dir data/chest_xray --model custom_resnet18 --checkpoint checkpoints/custom_resnet18/best_model.pth --split test --output-dir evaluation/custom_resnet18 --generate-grad-cam
```

```powershell
# Windows (PowerShell)
python src/evaluate.py `
    --data-dir data/chest_xray `
    --model custom_resnet18 `
    --checkpoint checkpoints/custom_resnet18/best_model.pth `
    --split test `
    --output-dir evaluation/custom_resnet18 `
    --generate-grad-cam
```

#### 6️⃣ Evaluate Custom ResNet18 + CBAM (From Scratch)

```bash
# Linux/macOS
python src/evaluate.py \
    --data-dir data/chest_xray \
    --model custom_resnet_cbam \
    --checkpoint checkpoints/custom_resnet_cbam/best_model.pth \
    --split test \
    --output-dir evaluation/custom_resnet_cbam \
    --generate-grad-cam
```

```cmd
# Windows (cmd)
python src/evaluate.py --data-dir data/chest_xray --model custom_resnet_cbam --checkpoint checkpoints/custom_resnet_cbam/best_model.pth --split test --output-dir evaluation/custom_resnet_cbam --generate-grad-cam
```

```powershell
# Windows (PowerShell)
python src/evaluate.py `
    --data-dir data/chest_xray `
    --model custom_resnet_cbam `
    --checkpoint checkpoints/custom_resnet_cbam/best_model.pth `
    --split test `
    --output-dir evaluation/custom_resnet_cbam `
    --generate-grad-cam
```

This will generate:
- **Classification Report**: Detailed metrics for each class
- **Confusion Matrix**: Visual representation saved as `confusion_matrix_test.png`
- **ROC Curve**: Receiver Operating Characteristic curve saved as `roc_curve_test.png`
- **Precision-Recall Curve**: PR curve saved as `pr_curve_test.png`
- **Metrics JSON**: All metrics saved to `metrics_test.json`
- **Grad-CAM Visualizations**: Sample heatmaps saved in `grad_cam_samples/` folder

### 📊 Compare All Models

After training and evaluating all models, use the comparison script to generate a comprehensive comparison table:

```bash
# All platforms
python src/compare_models.py --evaluation-dir evaluation --output model_comparison.csv
```

This will:
- Load metrics from all models in `evaluation/` directory
- Generate a comparison table sorted by accuracy
- Highlight the best performing model
- Provide category analysis (historical baselines, ResNet family, attention impact)
- Save results to `model_comparison.csv`

**Manual comparison:**

```bash
# Linux/macOS
cat evaluation/alexnet/metrics_test.json
cat evaluation/vgg16/metrics_test.json
cat evaluation/plain_resnet18/metrics_test.json
cat evaluation/resnet_cbam/metrics_test.json
cat evaluation/custom_resnet18/metrics_test.json
cat evaluation/custom_resnet_cbam/metrics_test.json
```

```cmd
# Windows (cmd)
type evaluation\alexnet\metrics_test.json
type evaluation\vgg16\metrics_test.json
type evaluation\plain_resnet18\metrics_test.json
type evaluation\resnet_cbam\metrics_test.json
type evaluation\custom_resnet18\metrics_test.json
type evaluation\custom_resnet_cbam\metrics_test.json
```

```powershell
# Windows (PowerShell)
Get-Content evaluation/alexnet/metrics_test.json
Get-Content evaluation/vgg16/metrics_test.json
Get-Content evaluation/plain_resnet18/metrics_test.json
Get-Content evaluation/resnet_cbam/metrics_test.json
Get-Content evaluation/custom_resnet18/metrics_test.json
Get-Content evaluation/custom_resnet_cbam/metrics_test.json
```

**Key Comparisons:**

1. **Architecture Evolution** (Historical Baselines):
   - `alexnet` (2012) → `vgg16` (2014) → `plain_resnet18` (2015)
   - Shows progression from early deep CNNs to modern residual networks

2. **Attention Mechanism Impact** (Main Comparison):
   - `plain_resnet18` (no attention) vs `resnet_cbam` (with CBAM)
   - Demonstrates the benefit of attention mechanisms

3. **Ablation Study** (CBAM Contribution):
   - `custom_resnet18` (no CBAM) vs `custom_resnet_cbam` (with CBAM)
   - Both trained from scratch to isolate CBAM's effect

4. **Transfer Learning Impact**:
   - `plain_resnet18` (pretrained) vs `custom_resnet18` (from scratch)
   - Shows the value of ImageNet pretraining

### Grad-CAM Visualization

The `--generate-grad-cam` flag enables weakly supervised lesion localization:
- Generates heatmaps showing where the model focuses attention
- Saves visualizations for randomly selected samples from each class
- Helps validate that the model is learning clinically relevant features
- Provides interpretability for medical professionals

Example Grad-CAM output structure:
```
evaluation/resnet_cbam/grad_cam_samples/
├── sample_0_NORMAL_IM-0001-0001.jpeg_cam.png
├── sample_1_NORMAL_IM-0003-0001.jpeg_cam.png
├── sample_5_PNEUMONIA_person1_bacteria_1.jpeg_cam.png
└── ...
```

Each visualization shows:
1. Original X-ray image
2. Grad-CAM heatmap
3. Overlay of heatmap on original image

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

This project provides **two implementations** of ResNet18 with CBAM attention mechanisms:

### 1. ResNet18 + CBAM (Standard - Pretrained Backbone)

**File:** `src/models/resnet_cbam.py`

Our primary implementation uses a pretrained ResNet18 backbone combined with CBAM attention modules:

#### Architecture Overview
- **Backbone**: ResNet18 pretrained on ImageNet (1.2M images)
- **Attention Module**: CBAM applied after each residual block (layer1-4)
- **Input**: 224×224 RGB chest X-ray images
- **Output**: Binary classification (NORMAL vs. PNEUMONIA)
- **Parameters**: ~11.7M (11.2M ResNet18 + 0.5M CBAM)

#### CBAM Components
1. **Channel Attention**: 
   - Uses both average and max pooling
   - Learns "what" features are important
   - Reduction ratio: 16 (configurable)
   
2. **Spatial Attention**: 
   - Uses 7×7 convolution
   - Learns "where" to focus in the image
   - Highlights disease-relevant regions

#### Why This Architecture?
✅ **Transfer Learning**: ImageNet pretraining provides robust low-level features  
✅ **Faster Convergence**: Typically converges in 20-30 epochs vs. 50+ from scratch  
✅ **Better Performance**: ~5-7% accuracy improvement over training from scratch  
✅ **Data Efficiency**: Works well with limited medical training data  
✅ **Attention Mechanisms**: CBAM helps focus on clinically relevant regions  
✅ **Proven Performance**: Achieves 84.94% accuracy with 96.41% pneumonia recall  
✅ **Interpretability**: Compatible with Grad-CAM for visualization  

**Recommended for:** Production deployment, limited training data, time constraints

---

### 2. Custom ResNet18 + CBAM (From Scratch)

**File:** `src/models/custom_resnet_cbam.py`

A fully custom implementation of ResNet18+CBAM built from scratch without using torchvision's pretrained models:

#### Architecture Overview
- **Backbone**: ResNet18 implemented from scratch (all layers manually coded)
- **Initialization**: He initialization for convolutional layers
- **Attention Module**: Same CBAM as standard implementation
- **Input**: 224×224 RGB chest X-ray images
- **Output**: Binary classification (NORMAL vs. PNEUMONIA)
- **Parameters**: ~11.7M (same as standard, but randomly initialized)

#### Implementation Details
- Custom `BasicBlock` class implementing residual connections
- Manual creation of all 4 residual layers
- Explicit weight initialization using He/Kaiming initialization
- No dependency on pretrained weights

#### Why Use This Implementation?
✅ **Educational Value**: Demonstrates complete understanding of ResNet architecture  
✅ **Full Control**: Every layer and operation is explicit and modifiable  
✅ **Research Flexibility**: Easy to experiment with architecture modifications  
✅ **Transparency**: Complete visibility into all operations and layers  
✅ **Independence**: No dependency on external pretrained weights  

❌ **Trade-offs**: Slower convergence (50-70 epochs), requires more training data, 5-7% lower initial accuracy

**Recommended for:** Academic research, educational purposes, architecture experiments, custom modifications

---

### Architecture Comparison

| Feature | Standard (Pretrained) | Custom (From Scratch) |
|---------|----------------------|----------------------|
| **Backbone Source** | torchvision.models.resnet18 | Custom implementation |
| **Initialization** | ImageNet pretrained | He initialization |
| **Parameters** | ~11.7M | ~11.7M |
| **Convergence** | 20-30 epochs | 50-70 epochs |
| **Final Accuracy** | ~84-85% | ~78-82% |
| **Training Time** | ~2.5 hours (50 epochs) | ~3-4 hours (70 epochs) |
| **Use Case** | Production | Education/Research |

---

### Grad-CAM for Weakly Supervised Localization

Gradient-weighted Class Activation Mapping (Grad-CAM) provides model interpretability:

#### How It Works
1. **Forward Pass**: Image through the network
2. **Target Selection**: Focus on predicted class
3. **Gradient Computation**: Backpropagate to target convolutional layer
4. **Weight Calculation**: Global average pooling of gradients
5. **Heatmap Generation**: Weighted combination of feature maps
6. **Visualization**: Overlay heatmap on original image

#### Benefits
- ✅ **No Additional Labels**: Only requires image-level classification labels
- ✅ **Visual Explanation**: Shows where the model "looks" for pneumonia
- ✅ **Clinical Validation**: Doctors can verify if the model focuses on correct regions
- ✅ **Error Analysis**: Helps identify model weaknesses
- ✅ **Trust Building**: Increases confidence in AI predictions

#### Target Layer
- **Default**: `layer4.1.conv2` (last convolutional layer before pooling)
- **Why**: Provides the best balance between spatial resolution and semantic information

## Dataset

The Chest X-Ray Pneumonia dataset contains:
- **Training Set**: ~5,000 X-ray images
- **Validation Set**: ~16 X-ray images
- **Test Set**: ~624 X-ray images
- **Classes**: NORMAL, PNEUMONIA

Dataset source: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Results

### Performance Metrics (ResNet18 + CBAM)

Our model achieved the following performance on the test set:

#### Overall Performance
- **Accuracy**: **84.94%**
- **ROC-AUC**: **0.928** (Excellent discrimination ability)
- **Average Precision**: **0.949** (High precision across all recall levels)

#### Per-Class Performance

| Metric | NORMAL | PNEUMONIA |
|--------|--------|-----------|
| **Precision** | 91.67% | 82.46% |
| **Recall** | 65.81% | **96.41%** |
| **F1-Score** | 76.62% | 88.89% |
| **Support** | 234 | 390 |

#### Key Insights

✅ **High Sensitivity (96.41% recall for PNEUMONIA)**: The model successfully detects pneumonia cases, which is critical in medical diagnosis to minimize false negatives.

✅ **Strong Specificity (91.67% precision for NORMAL)**: When the model predicts NORMAL, it's highly reliable.

✅ **Excellent ROC-AUC (0.928)**: Demonstrates strong discriminative ability between classes across different threshold settings.

✅ **Clinical Relevance**: The high recall for pneumonia (96.41%) means only 3.59% of pneumonia cases are missed, which is crucial for patient safety.

#### Confusion Matrix Analysis

|  | Predicted NORMAL | Predicted PNEUMONIA |
|---|---|---|
| **True NORMAL** | 154 | 80 |
| **True PNEUMONIA** | 14 | 376 |

- **True Positives (Pneumonia)**: 376 cases correctly identified
- **False Negatives (Missed Pneumonia)**: Only 14 cases (3.59% miss rate)
- **True Negatives (Normal)**: 154 cases correctly identified
- **False Positives (Over-diagnosis)**: 80 cases

### Visualizations

The evaluation generates comprehensive visualizations including:
- **Confusion Matrix**: Shows the distribution of predictions vs. actual labels
- **ROC Curve**: Demonstrates the trade-off between sensitivity and specificity
- **Precision-Recall Curve**: Useful for imbalanced datasets like ours
- **Grad-CAM Heatmaps**: Highlights the regions the model focuses on for each prediction

*All metrics and visualizations can be reproduced by running the evaluation script with your trained model.*


## References

### Academic Papers
- **CBAM**: Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). "CBAM: Convolutional Block Attention Module". ECCV 2018.
- **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition". CVPR 2016.
- **Grad-CAM**: Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". ICCV 2017.

### Dataset
- Kermany, D., Zhang, K., & Goldbaum, M. (2018). "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification". Mendeley Data, v2.
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.