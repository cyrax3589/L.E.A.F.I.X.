<div align="center">

# 🌿 L.E.A.F.I.X.
**L**eaf **E**valuation & **A**rtificial **F**eature **I**nference **eX**pert

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-FF6F00.svg?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> *An AIoT-inspired, deep learning diagnostic tool that brings laboratory-grade plant disease detection directly to your screen.*

---

</div>

<details>
  <summary><b> Table of Contents</b> (Click to expand)</summary>
  
  1. [Overview](#-overview)
  2. [Key Features](#-key-features)
  3. [System Architecture](#-system-architecture)
  4. [Installation & Setup](#-installation-&-setup)
  5. [Dataset](#-dataset)
  6. [Future Improvements](#-future-improvements)
  
</details>

---

## Overview

**L.E.A.F.I.X.** is a cutting-edge Convolutional Neural Network (CNN) application designed to combat agricultural loss by providing instant, highly accurate plant disease diagnostics. 

By leveraging transfer learning via `MobileNetV2` and wrapping the inference engine in a highly customized, non-scrolling Streamlit dashboard, L.E.A.F.I.X. bridges the gap between complex machine learning models and end-user accessibility. It not only predicts the disease but immediately suggests probable, actionable agronomic treatments.

---

## Key Features

### Model Capabilities
- [x] **Transfer Learning:** Fine-tuned `MobileNetV2` base model pre-trained on ImageNet.
- [x] **Robust Generalization:** Extensive preprocessing and image augmentation (rotation, zooming, flipping) to prevent overfitting.
- [x] **Multi-Class Classification:** Capable of differentiating between multiple disease states and healthy control leaves across various plant species.


---

## System Architecture

| Component | Technology Used | Purpose |
| :--- | :--- | :--- |
| **Backend Engine** | TensorFlow / Keras | Model training, layers, and inference execution. |
| **Base Model** | MobileNetV2 | High-efficiency feature extraction optimized for edge devices. |
| **Data Handling** | NumPy, Pillow | Image array formatting and mathematical operations. |
| **Frontend Framework** | Streamlit | Rapid prototyping and reactive UI generation. |
| **Styling** | Custom HTML/CSS | Overriding default layouts for a single-page app experience. |

---

## Installation & Setup

Get L.E.A.F.I.X. running on your local machine in under 2 minutes.

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/LEAFIX.git](https://github.com/YOUR_USERNAME/LEAFIX.git)
cd LEAFIX
```

## 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
## 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 4. Run the Application
Make sure your compiled model (best_model.keras) is in the root directory, then execute:
```bash
streamlit run app.py
```

---

## Dataset
The dataset used is the PlantVillage dataset, containing images of plant leaves categorized into different diseases and healthy classes.
https://www.kaggle.com/datasets/emmarex/plantdisease

---

## UI

<img width="1919" height="1016" alt="Screenshot 2026-03-19 134601" src="https://github.com/user-attachments/assets/3f6bb4b9-90ad-4588-a8b2-70cf5762fef2" />


---
## Future Improvements
- Add Grad-CAM for model interpretability
- Improve UI design
- Deploy the application online
- Expand dataset for more plant species
