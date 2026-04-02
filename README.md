---
title: Wine Quality Classifier
emoji: 🍷
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🍷 Wine Quality Predictor (ANN Multi-class Classification)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

A professional deep learning pipeline for multi-class classification of wine quality. This project uses an Artificial Neural Network (ANN) to categorize wines into quality tiers based on their physicochemical properties.

---

## 📋 Project Overview

### 1. The Problem
Wine quality assessment often relies on expensive sensory tests. This project provides a computational alternative, using chemical profiles (acidity, sugar, pH, alcohol) to predict the quality score of Portguese "Vinho Verde" wine.

### 2. The Dataset
* **Source**: UCI Machine Learning Repository (Wine Quality Dataset).
* **Classes**: Multi-class (Quality scores mapped to Low/Medium/High).
* **Features**: 11 physicochemical inputs.

### 3. The Solution (ANN Architecture)
* **Architecture**: Deep Neural Network built with PyTorch.
* **Hidden Layers**:
  * Dense (128 units) → BatchNorm → ReLU → Dropout (0.4)
  * Dense (64 units) → BatchNorm → ReLU → Dropout (0.3)
  * Dense (3) → LogSoftmax (for multi-class probabilities).
* **Optimization**: Adam optimizer with Cross-Entropy Loss.

---

## 📈 Numerical Results & Performance

The model achieved highly stable performance across multi-class quality tiers:

| Metric | Value | Significance |
| :--- | :--- | :--- |
| **Accuracy** | **83.8%** | Strong multi-class predictive capability. |
| **Weighted F1** | **81.9%** | Robust handling of class distribution. |
| **Macro F1** | **48.4%** | Captures performance across distinct quality tiers. |
| **Loss** | **Minimizing** | Stable convergence over 100 epochs. |

---

## 🚀 Deployment Strategy

### Docker Hub / Hugging Face Spaces
This project is configured for **Dockerized deployment** (Port 7860).

1. **Local Build**:
   ```bash
   docker build -t wine-quality-app .
   ```
2. **Local Run**:
   ```bash
   docker run -p 8501:7860 wine-quality-app
   ```

### Standard Local Run
```bash
conda activate ml-env
streamlit run app.py
```

---

## 🛠️ GitHub Configuration & Workflow

```bash
# 1. Initialize & track models with LFS
git init
git lfs install
git lfs track "models/*.pkl" "charts/*.png"

# 2. Deploy to GitHub
git add .
git commit -m "feat: Initial professional deployment of Wine Quality classification model and analytics dashboard"
git branch -M main
git remote add origin git@github.com:Ashutosh-AIBOT/ann-wine-quality-classifier.git
git push -u origin main
```

---

**Developed by [Ashutosh-AIBOT](https://github.com/Ashutosh-AIBOT)**
