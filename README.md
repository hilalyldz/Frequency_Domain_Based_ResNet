# Frequency Domain Based ResNet for GAN Image Detection (Training Branch)

This repository contains a training pipeline for GAN image detection using frequency-based features.

⚠️ **Note:**
This branch is focused on **training only**. It is not intended for full evaluation or benchmarking pipelines.

---

## 🎯 Overview

This project trains a ResNet-based classifier to distinguish between real and GAN-generated images using:

* Spatial domain (RGB)
* Frequency domain (FFT)
* Wavelet-based features

The implementation includes preprocessing, feature extraction, training, and logging.

---

## ⚙️ Features

* ✅ FFT-based feature extraction
* ✅ Wavelet-based feature extraction
* ✅ ResNet backbone
* ✅ Data augmentation support
* ✅ Dataset caching (`.pt` files)
* ✅ Training logs and checkpoints
* ✅ GPU training support (SLURM script)

---

## 📁 Project Structure

```text
code/
  cycleGAN_dataset.py        # Dataset + preprocessing (FFT / Wavelet)
  GAN_Detection_Train.py     # Core training pipeline
  run_training.py            # Training launcher
  run_gpu_resnet.sh          # GPU (SLURM) execution script

datasets/                    # Dataset directory (user-provided)
model_resnet/                # Saved model checkpoints
resnet_log/                  # Training logs
training.log                 # Per-batch logging
```

---

## 📂 Dataset Structure

The dataset must follow this structure:

```text
datasets/
  real/
    satellite/
      train/
      test/
  fake/
    satellite/
      train/
      test/
```

* Real images → label = 1
* Fake images → label = 0
* Images are resized to 256×256 and center-cropped to 224×224 for training

---

## 🚀 Setup

```bash
git clone <your-repo-url>
cd Frequency_Domain_Based_ResNet
pip install -r requirements.txt
```

---

## 🏋️ Training

Run training with:

```bash
python code/run_training.py --dataset=CycleGAN --feature=fft --gpu-id=0
```

---

### 🔄 Supported Features

Inside training you can switch feature type:

* `--feature=image` → RGB input
* `--feature=fft` → Frequency domain (default)
* `--feature=wavelet` → Wavelet features

Feature extraction is implemented in:
`GANDataset` (see `GAN_Detection_Train.py`)

---

## 🧠 Feature Motivation

This project explores different feature representations for image forgery detection:

* **FFT (Frequency Domain):**
  GAN-generated images often exhibit characteristic artifacts in the frequency domain. FFT-based features help capture these inconsistencies, making them effective for GAN detection.

* **Wavelet Features:**
  Wavelet transforms are widely used in image forensics to capture localized frequency variations and structural inconsistencies. While commonly applied in tasks such as copy-move forgery detection, in this project they are also explored for detecting GAN-generated images.

These features are used as alternative representations for training the same classification model.


---

## ⚡ GPU Training (Cluster / SLURM)

You can run training on GPU using:

```bash
bash code/run_gpu_resnet.sh
```

This script:

* activates conda environment
* requests GPU resources
* launches training with FFT feature

Script reference:
`code/run_gpu_resnet.sh` 

---

## ⚡ GPU Setup (Optional)

To use GPU acceleration, make sure you install a CUDA-enabled version of PyTorch.

### Check CUDA availability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

### Install PyTorch with CUDA

Visit the official PyTorch website:

👉 https://pytorch.org/get-started/locally/

Example (CUDA 11.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Notes

* If CUDA is not available, the code will run on CPU
* GPU is automatically used if available (`--gpu-id` parameter)


---

## 📊 Outputs

After training:

* Models → `model_resnet/`
* Logs → `resnet_log/`
* Batch logs → `training.log` 
* Grad-CAM visualizations → `outputs/cam/`

Example output:

```
Epoch 3, Train Loss: 0.6030, Val Loss: 0.7147, Val Acc: 69.23%
```

---

## 🧠 How It Works

1. Images are loaded and labeled
2. Optional transformation:

   * FFT (frequency filtering)
   * Wavelet decomposition
3. Data is cached as `.pt` files
4. Model is trained using ResNet
5. Loss, accuracy, and logs are recorded

---

## ⚠️ Important Notes

* This branch is **training-focused only**
* Testing pipeline is not the primary focus here
* Dataset must exist under `./datasets/`
* Cached dataset files (`*.pt`) are automatically created
* Default dataset used: `satellite`

---

## ⚠️ Testing Script

The repository includes a testing script (`GAN_Detection_Test.py` and `run_test.py`).

However, this branch is primarily focused on **training experiments** (FFT and wavelet features).
The testing pipeline is not the main focus and may require additional adjustments for full evaluation.


---

## 📜 Acknowledgment

Based on original work by Xu Zhang (Columbia University), extended with frequency-domain and wavelet-based analysis.
