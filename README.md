
# NT Fetal Anomaly Detection  

## Overview  
This project implements **deep learning-based NT measurement** from fetal ultrasound images to assist in anomaly detection. It employs **U-Net for segmentation** and **EfficientNet for classification**, integrating transformers and GANs for dataset augmentation and enhanced robustness.  

## Features  
- **NT Segmentation**: U-Net, ResU-Net, Attention U-Net for boundary delineation  
- **NT Classification**: EfficientNet, Vision Transformers for thickness prediction  
- **Preprocessing**: CLAHE, Gaussian denoising, ROI extraction, artifact removal  
- **Generative Models**: GANs for synthetic data augmentation  
- **Optimization**: Dice Loss, BCE, K-fold validation, real-time inference tuning  
- **Deployment**: Optimized for edge devices, clinical ultrasound integration  

## Methodology  

### **1. Data Collection & Preprocessing**  
- **Dataset**: Clinical fetal ultrasound images (11–14 weeks gestation)  
- **Normalization**: Intensity standardization across ultrasound devices  
- **Enhancement**: Adaptive histogram equalization, wavelet denoising  
- **ROI Extraction**: Bounding box localization, patch-based segmentation  

### **2. Model Development**  
- **Segmentation**: U-Net, ResU-Net, Attention U-Net  
- **Classification**: EfficientNet, Vision Transformers  
- **Generative Models**: GANs for synthetic NT image augmentation  
- **Attention Mechanisms**: Transformer-based feature enhancement  

### **3. Training & Evaluation**  
- **Loss Functions**: Dice Loss + BCE (Segmentation), Weighted CE (Classification)  
- **Validation**: K-fold cross-validation, AUROC, Dice Score, IoU  
- **Efficiency**: Quantization, pruning, inference speed optimization  

### **4. Deployment Pipeline**  
- **Edge Optimization**: Model compression for real-time inference  
- **Clinical Validation**: Benchmarking with expert annotations  
- **Deployment Formats**: Web-based, API endpoints, ultrasound firmware integration  

## Installation  
```bash
git clone https://github.com/gupta-nu/NT-fetal-anomaly-detection.git
cd NT-fetal-anomaly-detection
pip install -r requirements.txt
```

## Usage  

### **Preprocess Images**  
```bash
python preprocess.py --input data/raw --output data/processed
```
### **Train Models**  
```bash
python train.py --model unet --epochs 50
```
### **Evaluate Models**  
```bash
python evaluate.py --model unet
```

## License  
MIT License
