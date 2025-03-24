# Automated Defect Detection using Convolutional Autoencoders (MVTec AD)

This project presents a Deep Learning-based solution for automated quality inspection of industrial products, leveraging the [MVTec Anomaly Detection Dataset (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads). The primary goal is to automate visual defect detection, minimizing human error and reducing operational costs in industrial production lines.

We specifically implemented a **Convolutional Autoencoder (CAE)** to identify surface anomalies in products, focusing on the **bottle** category from the MVTec AD dataset.

---

## Project Pipeline

### 1. Data Preparation & Exploratory Analysis
- Utilized the MVTec AD dataset, focusing on the bottle category:
  - 300 normal images (training).
  - 92 images with various defects (testing).
- Conducted statistical analysis (intensity histograms, texture patterns) revealing repetitive textures in normal images and irregularities in defective samples.

### 2. Preprocessing
- Resized images to **256×256 pixels**.
- Normalized pixel values to range **[0,1]**.
- Applied **Data Augmentation**:
  - Random rotations.
  - Horizontal flips.
  - Brightness adjustments.

### 3. Dataset Splitting
- **Training**: Normal images only.
- **Testing**: Both normal and defective images, with no explicit labels for precise segmentation, motivating unsupervised learning.

### 4. Model Architecture
A **Convolutional Autoencoder** designed with:
- **Encoder**:
  - 3 convolutional layers (3×3 filters, stride=2, ReLU activation).
- **Latent Space**:
  - Compressed feature vector of dimension **64**.
- **Decoder**:
  - 3 deconvolutional layers with ReLU activation to reconstruct original resolution.

### 5. Training Configuration
- **Loss function**: Mean Squared Error (MSE).
- **Optimizer**: Adam, learning rate = 0.0002.
- **Batch size**: 32.
- **Epochs**: 200.
- Trained exclusively on normal samples.

### 6. Inference & Defect Detection
- The trained CAE reconstructs test images.
- **Error maps** generated as:  
  `Error = |Original Image - Reconstructed Image|`
- Higher intensity regions in the error map highlight potential defects.

### 7. Evaluation
- **Reconstruction MSE**: 0.0112
- **Peak Signal-to-Noise Ratio (PSNR)**: 19.95 dB
- Visual inspections confirm that defective regions are effectively highlighted in error maps.

---

## Key Results

| Metric                    | Value        |
|--------------------------|-------------|
| Reconstruction Error (MSE) | **0.0112**  |
| PSNR                     | **19.95 dB** |

Visual outputs demonstrate clear differentiation between normal and defective areas, validating the effectiveness of the CAE model.

---

## Limitations & Future Improvements

### Identified Limitations:
- Subtle defects resembling background patterns may not be detected.
- Threshold definition for binarizing error maps is sensitive and fixed.

### Proposed Enhancements:
1. Integrate **Attention Mechanisms** to focus on potential defect regions.
2. Explore **Variational Autoencoders (VAEs)** or **U-Net Autoencoders** to enhance model expressiveness.
3. Apply morphological post-processing to reduce false positives.
4. Implement adaptive thresholding for error maps.
5. Expand experiments to other categories within the MVTec AD dataset to assess robustness.

---
