# 🌊 Wavelet-Based GAN Image Detection (Testing Branch)

This repository provides a **testing and analysis pipeline** for CMF image detection using **wavelet-based features**.

⚠️ **Important Note**
This branch is **not intended for training**.
It is specifically designed to **evaluate a pre-trained model** that was trained using wavelet representations.

---

## 🎯 Purpose of This Branch

The goal of this branch is to:

* Evaluate a **ResNet-based classifier** trained on **wavelet-transformed images**
* Analyze model decisions using **Grad-CAM**
* Perform **wavelet-domain explainability**
* Project frequency-domain activations back to the **spatial domain**

---

## 🧠 Key Idea

Instead of using raw RGB images, the model operates on:

> **12-channel wavelet decomposition (LL, LH, HL, HH for each RGB channel)**

This enables the model to:

* Capture **frequency artifacts**
* Detect **GAN-specific inconsistencies**
* Provide **interpretable frequency-based explanations**

---

## ⚙️ Features

### ✅ Wavelet-Based Inference

* Input: 12-channel wavelet tensor
* Wavelet type: `haar`
* Channels:

  * LL (approximation)
  * LH, HL, HH (detail components)

---

### 🔥 Grad-CAM Explainability

* Applied on the last ResNet layer
* Produces activation maps in **frequency domain**

---

### 🔄 Spatial Backprojection

* Wavelet coefficients are weighted using Grad-CAM
* Inverse wavelet transform reconstructs:

> 📍 **Artifact localization map in image space**

---

### 📊 Band Contribution Analysis

For each image:

* Computes contribution of:

  * LL (low frequency)
  * LH / HL (edge structures)
  * HH (high-frequency noise)

Also reports:

* Dominant frequency band
* Average contributions (real vs fake)

---

### 🖼️ Saved Outputs

For each test image:

* Grad-CAM heatmap
* Spatial projection map
* Overlay visualization
* Colorbar-enhanced figures (for thesis usage)
* Band contribution report (`.txt`)

---

## 📂 Project Structure

```
.
├── code/
│   ├── GAN_Detection_Test.py      # Main testing pipeline
│   ├── run_test.py                # Experiment runner
│   └── cycleGAN_dataset.py        # Dataset loader
│
├── datasets/
│   ├── real/
│   └── fake/
│
├── model_resnet/
│   └── <trained_model_checkpoints>
│
├── outputs/
│   └── spectral_explainability/
│
├── logs_test/
└── requirements.txt
```

---

## 📦 Dataset Format

Dataset should be organized as:

```
datasets/
├── real/
│   └── satellite/
│       ├── train/
│       └── test/
└── fake/
    └── satellite/
        ├── train/
        └── test/
```

* Image size: automatically resized to **256×256**
* Format: `.jpg`

Dataset loading and caching is handled by:
👉 

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:

* PyTorch
* torchvision
* OpenCV
* PyWavelets
* Grad-CAM

(See full list: )

---

### 2. Run Testing

```bash
python run_test.py --dataset=CycleGAN --feature=wavelet
```

This will:

* Load trained model
* Run inference on test set
* Generate visual explanations
* Save results

Execution pipeline:
👉 

---

## 🧪 Model Details

* Backbone: **ResNet-34**
* Modified first layer:

  * Input channels: **12 (wavelet)**
* Output:

  * Binary classification (real vs fake)

Model loading:

* Pretrained weights are loaded from:

```
./model_resnet/<experiment_name>/checkpoint_X.pth
```

---

## 📈 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Classification Report (precision, recall, F1)

All metrics are:

* Printed to console
* Saved to file

---

## 🔍 Explainability Pipeline

1. Forward pass → prediction
2. Grad-CAM → frequency importance
3. Weight wavelet coefficients
4. Inverse transform → spatial artifact map

Implementation:
👉 

---

## ⚠️ Limitations

* Assumes **dataset order consistency** between:

  * cached tensors
  * raw images (for visualization)

* This branch is:

  * ❌ Not optimized for training
  * ❌ Not a benchmarking pipeline
  * ✅ Focused on analysis & visualization

---

## 📌 Notes

* Designed for **research & thesis experiments**
* Visualization outputs are suitable for:

  * Papers
  * Presentations
  * Qualitative analysis

---

## 🙌 Acknowledgment

This project is based on:

* AutoGAN framework
* Frequency-domain fake image detection research

---

## 📬 Author

Hilal Yildiz
