# Frequency Domain Based ResNet (FFT Test Branch)

This branch focuses on **testing a GAN image detector trained with FFT (frequency-domain) features**.

⚠️ **Important**

* This branch is **for inference/testing only**
* Training pipeline is not the focus here
* Only **FFT-based models** are supported

---

## 🎯 Overview

This project evaluates a ResNet-based GAN detector using **frequency-domain representations (FFT)**.

The pipeline:

1. Load pre-trained model
2. Apply FFT-based feature processing
3. Perform classification (real vs fake)
4. Generate **Grad-CAM visualizations in frequency domain**
5. Project explanations back to spatial domain

---

## 📂 Project Structure

```
Frequency_Domain_Based_ResNet/
│
├── code/
│   ├── GAN_Detection_Test.py
│   ├── GAN_Detection_Train.py
│   ├── cycleGAN_dataset.py
│   ├── run_test.py
│   ├── Utils.py
│
├── datasets/
├── model_resnet/
├── outputs/
│   └── explainability/
│
├── logs_test/
├── README.md
├── requirements.txt
```

---

## 📊 Dataset Structure

The dataset must follow this structure:

```
datasets/
├── real/
│   └── satellite/
│       └── test/
│           ├── img1.jpg
│           └── ...
├── fake/
│   └── satellite/
│       └── test/
│           ├── img1.jpg
│           └── ...
```

The dataset loader caches processed data as `.pt` files for faster loading .

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🔹 Run FFT-based testing

```bash
python code/run_test.py --feature=fft
```

This will:

* Load the trained model
* Run inference on the dataset
* Save logs automatically

Command internally executed :

```bash
python code/GAN_Detection_Test.py \
    --training-set satellite \
    --model=resnet \
    --test-set=transposed_conv \
    --feature=fft \
    --model-dir ./model_resnet/
```

---

## 🧠 Model

* Backbone: **ResNet34**
* Output: Binary classification (real / fake)
* Checkpoints loaded from:

```
model_resnet/<experiment_name>/checkpoint_XX.pth
Exp:
model_resnet/satellite_da_fft_0_resnet/checkpoint_20.pth

```

---

## 🔬 Explainability (Grad-CAM)

This branch includes advanced explainability:

### ✔ Frequency-domain Grad-CAM

* Highlights important frequency regions

### ✔ Spatial Backprojection

* Converts frequency importance → spatial artifact map

### ✔ Band Analysis

* Low / Mid / High frequency contribution

Outputs are saved to:

```
outputs/explainability/
```

Including:

* Grad-CAM heatmaps
* Spatial projections
* Overlay visualizations
* Band statistics (TXT + CSV)
* Plots (histograms, bar charts)

---

## 📈 Metrics

The following are computed:

* Accuracy
* TPR (True Positive Rate)
* TNR (True Negative Rate)
* Confusion Matrix
* Classification Report

---

## ⚠️ Notes

* This branch assumes **pre-trained FFT models already exist**
* Some components (e.g., wavelet) are present but not the focus
* Data loading uses cached `.pt` files for efficiency

---

## 🔧 Known Limitations

* Dataset indexing between loaders and raw images is not strictly guaranteed
* No unified config system (CLI-based)
* Limited dataset generalization (currently satellite)

---

## 📌 Future Improvements

* Config-based experiment management
* Full reproducibility pipeline
* Multi-dataset evaluation
* Cleaner separation of training vs inference

---

## ✨ Summary

This branch provides a **clean and portable pipeline** for:

👉 Testing FFT-based GAN detectors 
👉 Visualizing frequency-domain artifacts
👉 Analyzing frequency contributions

---

## 👤 Author

Hilal Yıldız
