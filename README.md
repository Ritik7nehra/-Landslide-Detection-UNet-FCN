# -Landslide-Detection-UNet-FCN
A deep learning-based system for landslide detection using multi-spectral satellite imagery (NDVI, DEM, Slope, RGB) with U-Net and FCN architectures for precise image segmentation and disaster prediction.

# Landslide Detection with Multi-Spectral Satellite Imagery and Image Segmentation

This repository contains a deep learning-based approach for **landslide detection** using **multi-spectral satellite imagery**, integrating features such as **NDVI**, **DEM**, **Slope**, and **RGB** data. The system employs **U-Net** and **Fully Convolutional Networks (FCN)** for pixel-level segmentation to identify landslide-affected regions.

## 🌍 Overview
Traditional landslide detection methods are time-consuming and inefficient. This project leverages remote sensing and deep learning for automated, accurate, and scalable landslide prediction.

## 🧠 Models Used
- **U-Net:** Performs pixel-wise segmentation with high boundary accuracy.
- **FCN:** Provides full-image segmentation for coarse detection.
- Both models are evaluated on metrics like **Precision**, **Recall**, and **F1 Score**.

## 📊 Results Summary
| Model | Precision | Recall | F1-Score |
|--------|------------|---------|-----------|
| U-Net | 0.78 | 0.68 | 0.72 |
| FCN | 0.73 | 0.64 | 0.68 |

## ⚙️ Features
- Multi-spectral input (NDVI, DEM, Slope, RGB)
- Data preprocessing and augmentation
- Deep learning segmentation using TensorFlow/Keras
- Visualization of predicted landslide regions

## 🧩 Requirements
Install dependencies via:
```bash
pip install -r requirements.txt

