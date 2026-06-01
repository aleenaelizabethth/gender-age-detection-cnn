# 🧠 Gender & Age Detection System
### MobileNetV2-based Deep Learning Model for Real-time Demographic Classification

> A multi-output deep learning model using **MobileNetV2** backbone trained on the UTKFace dataset to simultaneously predict **gender** and **age** from facial images — with real-time **webcam detection** support.

---

## 🎬 Demo Videos

▶️ **[Test on Photos](test_on_photos.mp4)** — Model predicting gender & age on static images

▶️ **[Live Webcam Detection](live_web_cam.mp4)** — Real-time detection via webcam feed

---

## 🔍 Project Overview

This project builds a deep learning pipeline that detects faces and predicts:
- 👤 **Gender** — Male or Female (binary classification)
- 🎂 **Age** — Estimated age in years (regression)

Both predictions are made simultaneously using a **MobileNetV2 pretrained backbone** with two output heads, combined with **Haar Cascade face detection** for real-time performance.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

---

## 📁 Project Structure

```
gender-age-detection-cnn/
│
├── model.ipynb                  # Full pipeline — data loading, training & inference
├── best_model_weights.h5        # Saved trained model weights
├── test_on_photos.mp4           # Demo — prediction on static images
├── live_web_cam.mp4             # Demo — real-time webcam detection
└── README.md
```

---

## 🔄 Workflow

### 1. Dataset
- **UTKFace Dataset** — 20,000+ face images from [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- Filenames encode labels: `age_gender_ethnicity_date.jpg`
- Images resized to **64×64 pixels**
- Age range: **1–116 years**

### 2. Data Preprocessing
- Parsed age and gender labels from filenames
- Converted BGR → RGB using OpenCV
- Normalised pixel values to **[0, 1]**
- Normalised age to **[0, 1]** for stable training
- **80/20 train-validation split**

### 3. Model Architecture
- **MobileNetV2 pretrained backbone** (ImageNet weights, frozen)
- GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.4)
- Two output heads:
  - `gender` → sigmoid activation (binary classification)
  - `age` → linear activation (regression)
- Optimiser: **Adam** (lr=1e-4)
- Loss: `binary_crossentropy` (gender) + `MAE` (age)

### 4. Face Detection & Inference
- **Haar Cascade** face detector (with YOLO fallback)
- Prediction smoothing using a **rolling buffer of 20 frames** for stable real-time output
- Displays: Gender + Age + Confidence % on bounding box

### 5. Real-time Webcam
- Live webcam feed with face detection
- Smoothed predictions updated every frame
- Press **Q** to quit

---

## ▶️ How to Run

```bash
# Install dependencies
pip install tensorflow opencv-python numpy scikit-learn

# Download UTKFace dataset from Kaggle and place in:
# data/UTKFace/

# Open and run the notebook
jupyter notebook model.ipynb
```

---

## 📊 Model Output

| Output | Type | Activation | Metric |
|---|---|---|---|
| Gender | Binary Classification | Sigmoid | Accuracy |
| Age | Regression | Linear | MAE |

---

## 👩‍💻 Author

**Aleena Elizabeth Thomas**  
[LinkedIn](https://linkedin.com/in/aleena-elizabeth-thomas) · [GitHub](https://github.com/aleenaelizabethth)
