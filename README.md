# Caltech-256 Image Classification (Traditional ML + End-to-End Fine-tuning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zdhmPLfFrvzzH1dQLgOajEjJOpzQtdhH?usp=sharing)

Colab URL: https://colab.research.google.com/drive/1zdhmPLfFrvzzH1dQLgOajEjJOpzQtdhH?usp=sharing

**Course:** Học Máy (Machine Learning) — HCMUT  
**Term:** Học kỳ `252` (Năm học 2025–2026)  
**Instructor:** Dr. Trương Vĩnh Lân  
**Project:** Caltech-256 (Image Data) — Group 4

## Team Members & Work Distribution

| No. | Student Name | Tasks & Responsibilities | Contribution |
|---:|---|---|---:|
| 1 | Lê Ngọc Hùng Dũng | Dataset Selection, Exploratory Data Analysis (EDA), ML | 100% |
| 2 | Nguyễn Hồ Khanh | ML, Deep Learning, Fine-tuning, Experiment Grid | 100% |
| 3 | Lê Hoàng Chí Vĩ | Feature Extraction, Feature Management | 100% |
| 4 | Trương Minh Quân | Report, Writing, Code Packaging | 100% |
| 5 | Trần Văn Hùng | ML, Deep Learning, Fine-tuning, Experiment Grid | 100% |

## Overview

This repository implements both:

- A **traditional machine-learning pipeline** (deep feature extraction → classical classifiers)
- A **deep learning pipeline** with **end-to-end fine-tuning** of pretrained models

Main notebook: `Group_4_ML_Assignment___Image_Data.ipynb`

### Traditional ML pipeline (required)

1. Basic EDA / dataset inspection
2. Data preprocessing (resize/crop/normalize)
3. **Deep feature extraction** using pretrained vision models (e.g., ResNet/EfficientNet/ViT)
4. Train & evaluate traditional classifiers (Logistic Regression / Linear SVM / Random Forest, etc.)

### Deep learning pipeline (extension)

We additionally conduct **end-to-end fine-tuning** (transfer learning) on Caltech-256 using:

- ResNet18

## Project Structure

```
.
|-- Group_4_ML_Assignment___Image_Data.ipynb
|-- features/              # generated feature files (.h5, .npy)
|-- notebooks/             # optional place to store notebooks
`-- reports/               # optional place to store the PDF report
```

## Setup (Windows + CUDA) with `uv`

This project uses `uv` to manage Python + dependencies.

```powershell
cd "d:\Hung-Dung\HCMUT\252\Machine Learning\Caltech-ML-Assignment"
uv sync
```

Quick CUDA check:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Run the Notebook

```powershell
uv run jupyter lab
```

Then open `Group_4_ML_Assignment___Image_Data.ipynb` and run **Kernel → Restart Kernel and Run All Cells**.


