# Project Report: Chest X-Ray Pneumonia Classification

**Author:** CHEN Xinyu  
**Student ID:** 23270217    
**GitHub:** https://github.com/Leoxenon/Chest-X-Ray-Pneumonia-Classification

## 1. Introduction

Pneumonia causes over 800,000 child deaths annually (WHO, 2023), making accurate chest X-ray diagnosis critical. However, radiologist shortages and diagnostic variability (60-80% inter-observer agreement) create challenges. This project aims to discover an effective way to realize automated pneumonia detection using deep learning.

**Dataset:** Chest X-Ray Images from Kaggle (Kermany et al., 2018) - 5,216 training images, 624 test images (234 normal, 390 pneumonia) from pediatric patients aged 1-5.

**Objective:** Discover the most suitable baseline model and verify whether the newly proposed creative features are genuinely effective in current task.

## 2. Methodology

### 2.1 Baseline Model Comparison

To scientifically justify the architecture choice, I trained and compared six models representing CNN evolution from 2012-2018:

**Historical Baselines:**
These baselines utilize pretrained models provided by PyTorch. 
- **AlexNet (2012)**: First deep CNN breakthrough, 8 layers, ~57M parameters
- **VGG16 (2014)**: Deep uniform architecture with 3×3 filters, 16 layers, ~138M parameters
- **Plain ResNet18 (2015)**: Introduced skip connections, 18 layers, ~11M parameters

**Proposed Models:**
After Comparing baseline models, I selected ResNet18 for further exploration. Besides PyTorch pretrained ResNet18, I also construct a ResNet18 from scratch.
- **ResNet18-CBAM (Pretrained)**: ResNet18 + attention mechanisms with ImageNet initialization
- **Custom ResNet18 (From Scratch)**: Baseline without attention (ablation control)
- **Custom ResNet18-CBAM (From Scratch)**: ResNet18 + attention from random initialization (ablation test)

**Training Setup:**
- Loss: Weighted Cross-Entropy (class weights: [0.74, 0.26] to handle imbalance)
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Augmentation: Random flips, rotation (±10°), ColorJitter
- Epochs: 50 (pretrained models), 70 (from-scratch models)
- Batch Size: 32 (ResNet/AlexNet), 16 (VGG16 due to 138M parameters)

### 2.2 Innovation 1: CBAM Attention Mechanism

**Why CBAM?** Standard CNNs treat all features equally. CBAM (Woo et al., 2018) adds attention to focus on relevant regions:

```
Feature Map → Channel Attention (what is important?)
           → Spatial Attention (where to focus?)
           → Refined Features
```

- **Channel Attention**: Uses avg/max pooling + MLP to weight feature channels
- **Spatial Attention**: Uses 7×7 conv on channel-pooled features to create spatial weights
- **Cost**: Only +0.5M parameters (~4% overhead on 11M ResNet18)

### 2.3 Innovation 2: Grad-CAM Interpretability

**Why Grad-CAM?** Medical AI requires trust. Grad-CAM (Selvaraju et al., 2017) visualizes where the model focuses:

```
Target Class → Compute Gradients → Weight Feature Maps → Generate Heatmap
```

- Highlights lung opacities, infiltrates, and consolidations in pneumonia cases
- Enables radiologists to verify model reasoning
- Critical for clinical deployment and regulatory approval

**Implementation:** Target layer = `layer4` (final conv layer before pooling)

## 3. Results and Analysis

### 3.1 Baseline Comparison: Why ResNet18?

**Test Set Performance (624 images):**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Parameters |
|-------|----------|-----------|--------|----------|---------|------------|
| **VGG16** | **88.46%** | 86.64% | 96.41% | 91.26% | **95.94%** | 138M |
| **Plain ResNet18** | **87.98%** | 85.55% | **97.18%** | **91.00%** | **96.25%** | 11M |
| **AlexNet** | 85.74% | 84.28% | 94.87% | 89.26% | 94.49% | 57M |

**Key Findings:**

1. **Architecture Evolution Validated:**
   - VGG16 (2014) achieves highest accuracy (88.46%) but has **12× more parameters** than ResNet18
   - Plain ResNet18 (2015) achieves comparable performance (87.98%) with **only 11M parameters**
   - AlexNet (2012) shows expected lower performance (85.74%) with outdated architecture

2. **ResNet18 Chosen for Efficiency-Performance Trade-off:**
   - **Similar performance to VGG16** (87.98% vs 88.46%, only -0.48%)
   - **12× fewer parameters** (11M vs 138M) → faster inference, less memory
   - **Best ROC-AUC** (96.25%) among all models → superior discrimination ability
   - **Highest recall** (97.18%) → critical for medical screening (minimize false negatives)

3. **Why Not VGG16?** Despite slightly higher accuracy:
   - Requires 16 batch size (vs 32 for ResNet18) due to memory constraints
   - 3× slower training time
   - Impractical for deployment on resource-limited devices

### 3.2 Ablation Study: Does CBAM Help?

**Comparing Plain ResNet18 vs ResNet18-CBAM (both pretrained):**

| Metric | Plain ResNet18 | ResNet18-CBAM | Δ (Impact of CBAM) |
|--------|----------------|---------------|---------------------|
| Accuracy | **87.98%** | 84.94% | **-3.04%** ⚠️ |
| Recall | **97.18%** | 96.41% | -0.77% |
| Precision | **85.55%** | 82.46% | -3.09% |
| ROC-AUC | **96.25%** | 92.81% | **-3.44%** ⚠️ |

**Unexpected Result:** CBAM **decreased** performance on pretrained ResNet18!

**Analysis:**
- Pretrained ResNet18 already learned robust features from ImageNet
- Adding CBAM may **interfere with pretrained weights** during fine-tuning
- Attention modules initialized randomly while backbone is pretrained → training instability
- **Lesson:** Attention mechanisms don't always improve pretrained models

**Comparing Custom ResNet18 vs Custom ResNet18-CBAM (both from scratch):**

| Metric | Custom ResNet18 | Custom ResNet18-CBAM | Δ (Impact of CBAM) |
|--------|-----------------|----------------------|---------------------|
| Accuracy | **83.65%** | 81.73% | **-1.92%** ⚠️ |
| Recall | 96.92% | **98.72%** | **+1.80%** ✓ |
| Precision | **80.77%** | 77.94% | -2.83% |
| ROC-AUC | **93.76%** | 93.33% | -0.43% |

**Mixed Results:**
- CBAM improves **recall** (+1.80%) → better pneumonia detection
- But hurts **accuracy** (-1.92%) and **precision** (-2.83%) → more false positives
- Trade-off: Higher sensitivity at cost of specificity

**Conclusion on CBAM:**
Generally, The performance with CBAM is reduced for both pretrained and custom models. The pretrained model was more significantly affected. While the customized model's overall performance saw a slight decline, its sensitivity improved.
- ✓ **Benefit for from-scratch training:** Improves recall (critical for medical screening)
- ✗ **Hurts pretrained models:** Interferes with existing features
- ⚠️ **Trade-off:** Better sensitivity, worse specificity
- **Decision:** Use plain ResNet18 for this dataset (better overall balance)

### 3.3 Grad-CAM Interpretability Results

**Purpose:** Visualize model attention to verify clinical relevance

**Observations:**
- ✓ **Pneumonia cases:** Heatmaps highlight lung infiltrates, opacities, and consolidations
- ✓ **Normal cases:** Distributed, low-intensity activations across lung fields
- ⚠️ **Some false positives:** Activations on cardiac silhouette edges (not pneumonia)
- ⚠️ **Subtle cases:** Early-stage pneumonia with faint findings less consistently detected

**Representative Grad-CAM Examples (ResNet18-CBAM):**

<div align="center">
  <img src="evaluation/resnet_cbam/grad_cam_visualizations/grad_cam_sample_0_IM-0001-0001.jpeg" width="45%" alt="Normal Case 1"/>
  <img src="evaluation/resnet_cbam/grad_cam_visualizations/grad_cam_sample_2_IM-0005-0001.jpeg" width="45%" alt="Normal Case 2"/>
  <p><em>Figure 1: Normal chest X-rays with diffuse, low-intensity attention across lung fields (True Negatives)</em></p>
</div>

<div align="center">
  <img src="evaluation/resnet_cbam/grad_cam_visualizations/grad_cam_sample_5_person100_bacteria_475.jpeg" width="45%" alt="Pneumonia Case 1"/>
  <img src="evaluation/resnet_cbam/grad_cam_visualizations/grad_cam_sample_6_person100_bacteria_477.jpeg" width="45%" alt="Pneumonia Case 2"/>
  <p><em>Figure 2: Bacterial pneumonia cases with focused attention on lung infiltrates and opacities (True Positives)</em></p>
</div>

**Clinical Value:**
- Radiologists can verify model focuses on relevant anatomical regions
- Builds trust by showing "reasoning" (not black box)
- Helps identify model weaknesses (edge artifacts, cardiac shadow confusion)

## 4. Discussion

### 4.1 Summary of Findings

**1. ResNet18 Justified Through Systematic Comparison**
- Baseline comparison validates ResNet18 as optimal choice
- Achieves **97.18% recall** (best among all models) with only **11M parameters**
- VGG16 slightly better accuracy (+0.48%) but **12× more parameters** → impractical
- AlexNet outdated architecture shows expected lower performance (-2.24% accuracy)

**2. CBAM Attention: Context-Dependent Value**
- ✗ **Hurts pretrained ResNet18**: -3.04% accuracy, -3.44% ROC-AUC
  - Reason: Interferes with pretrained ImageNet features during fine-tuning
- ✓ **Helps from-scratch training**: +1.80% recall (better pneumonia detection)
  - Reason: Guides learning when starting from random weights
- Convolutional Block Attention Module (CBAM) may not help improve the performance for both models, especially harm the transfer learning. However, it can beneficial for from-scratch training in some scenarios where high sensitivity is the main objective.

**3. Grad-CAM Validates Clinical Relevance**
- Model correctly focuses on lung infiltrates and opacities in pneumonia
- Enables radiologist verification of predictions (critical for trust)
- Identifies failure modes: cardiac shadow confusion, edge artifacts

### 4.2 Limitations

**Dataset Constraints:**
- Single institution, pediatric-only (ages 1-5) → generalization to adults unproven
- Small validation set (16 images) → limited hyperparameter tuning confidence
- Class imbalance (74% pneumonia) may not reflect real-world prevalence

**Performance Gaps:**
- All models show false positive rates 13-22% on normal cases
- CBAM tradeoff: Higher sensitivity but lower specificity
- No multi-class classification (bacterial vs viral pneumonia)

### 4.3 Future Work

**Immediate:**
- Ensemble plain ResNet18 + ResNet18-CBAM to balance performance
- Threshold optimization for different use cases (screening vs diagnosis)
- Test on external datasets (NIH ChestX-ray14, CheXpert, MIMIC-CXR)

**Long-term:**
- Multi-institutional validation with adult patients
- Multi-class extension (bacterial/viral/COVID-19 pneumonia)
- Prospective clinical trial with radiologist feedback

---

## References

1. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
2. Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.
3. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV 2017*.
4. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses by Image-Based Deep Learning. *Cell*, 172(5).
5. Krizhevsky, A., et al. (2012). ImageNet Classification with Deep CNNs. *NIPS 2012*.
6. Simonyan, K., & Zisserman, A. (2015). Very Deep CNNs for Large-Scale Image Recognition. *ICLR 2015*.