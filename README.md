

# Anomalyze

## Overview

**Anomalyze** is a deep learning framework for **automated nuchal translucency (NT) measurement** from fetal ultrasound scans acquired during the **11–14 week gestational window**. NT thickness is a clinically validated biomarker for early detection of **chromosomal abnormalities (e.g., Trisomy 21), congenital heart defects, and other structural anomalies**.

Manual NT measurement is highly operator-dependent and subject to inter-observer variability. **EchoWomb standardizes this process through a fully automated pipeline**, integrating **segmentation, classification, and generative augmentation models** to achieve consistent, precise, and real-time NT assessment.

This repository provides the **complete workflow**: from image preprocessing and model training to evaluation, optimization, and deployment for clinical integration.



## ✨ Key Features

* **Segmentation**:

  * U-Net, ResU-Net, and Attention U-Net architectures for pixel-accurate delineation of NT boundaries.
* **Classification**:

  * EfficientNet and Vision Transformers for NT thickness-based risk stratification.
* **Preprocessing Pipeline**:

  * CLAHE, Gaussian/wavelet denoising, ROI extraction, and artifact suppression to normalize ultrasound inputs.
* **Generative Augmentation**:

  * GAN-based synthetic image generation to mitigate dataset scarcity and enhance model robustness.
* **Optimization & Evaluation**:

  * Hybrid loss functions (Dice + BCE), K-fold cross-validation, AUROC, IoU, and Dice Score.
* **Deployment**:

  * Model pruning, quantization, and compression for edge-device inference and real-time ultrasound integration.



## 📊 Methodology

### 1. Data Collection & Preprocessing

* **Dataset**: First-trimester fetal ultrasound scans (11–14 weeks).
* **Normalization**: Intensity standardization across heterogeneous ultrasound devices.
* **Enhancement**: Adaptive histogram equalization, Gaussian/wavelet denoising.
* **ROI Extraction**: Bounding-box localization of NT region and patch-based segmentation for fine detail.

### 2. Model Development

* **Segmentation Networks**:

  * U-Net baseline for medical image segmentation.
  * ResU-Net with residual skip connections for deeper feature learning.
  * Attention U-Net for contextual focus on NT boundaries.
* **Classification Networks**:

  * EfficientNet (lightweight, parameter-efficient backbone).
  * Vision Transformers (ViTs) for global contextual reasoning on NT thickness.
* **Generative Models**:

  * GANs trained on ultrasound distributions to augment rare NT cases.
* **Attention Mechanisms**:

  * Transformer-based feature refinement for improved classification sensitivity.

### 3. Training & Evaluation

* **Loss Functions**:

  * Dice Loss + BCE for segmentation stability.
  * Weighted Cross-Entropy for classification imbalance.
* **Validation Protocols**:

  * K-fold cross-validation.
  * Metrics: AUROC, IoU, Dice Score, Precision-Recall.
* **Efficiency Measures**:

  * Model pruning and quantization for faster inference.
  * Batch normalization and mixed-precision training.

### 4. Deployment Pipeline

* **Edge Optimization**:

  * Deployment-ready models compressed for bedside ultrasound machines.
* **Clinical Validation**:

  * Benchmarked against sonographer annotations to assess clinical reliability.
* **Deployment Formats**:

  * Web-based interface, REST APIs, or firmware-level integration into ultrasound devices.


## ⚙️ Installation

```bash
git clone https://github.com/gupta-nu/EchoWomb.git
cd EchoWomb
pip install -r requirements.txt
```


## 🚀 Usage

**Preprocess Images**

```bash
python preprocess.py --input data/raw --output data/processed
```

**Train Segmentation Models**

```bash
python train.py --model unet --epochs 50
```

**Evaluate Models**

```bash
python evaluate.py --model unet
```



## 🏥 Clinical Relevance

* **Why NT?**
  Nuchal translucency measurement in the first trimester is one of the most important non-invasive markers for **early prenatal anomaly screening**.
* **Impact of Automation**:
  EchoWomb reduces human error, accelerates workflows, and enhances **reproducibility of NT measurements**, enabling broader access to early screening in both advanced and resource-limited clinical settings.



## 📄 License

MIT License
