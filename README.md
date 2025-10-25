# 🧠 Easy Data Augmentation for Text Classification — SST-2 Reproduction

### Reproduction of _“EDA: Easy Data Augmentation Techniques for Text Classification” (Wei & Zou, 2019)_

> _Course Project — Generative AI / Machine Learning Lab (Milestone 2)_

---

## 📘 Overview

This repository reproduces the **Easy Data Augmentation (EDA)** approach for text classification using the **Stanford Sentiment Treebank v2 (SST-2)** dataset.  
The goal is to verify the original paper’s claim that simple text augmentations (synonym replacement, random insertion, swap, deletion) can improve model performance—especially when the amount of labeled data is small.

The project implements:

- **Custom augmentation logic** in [`01_augment_ml.ipynb`](./01_augment_ml.ipynb)
- **Model training & evaluation pipeline** in [`02_train.py`](./02_train.py)

---

## 🧩 Dataset

**Dataset:** [Stanford Sentiment Treebank v2 (SST-2)](https://nlp.stanford.edu/sentiment/)  
**Task:** Binary sentiment classification (Positive / Negative)

| Split        | Portion of Training Set | Approx Samples |
| :----------- | :---------------------: | :------------: |
| `1_tiny`     |           5 %           |    ≈ 3 000     |
| `2_small`    |          20 %           |    ≈ 13 000    |
| `3_standard` |          50 %           |    ≈ 33 000    |
| `4_full`     |          100 %          |    ≈ 67 000    |

All models are evaluated on the same fixed SST-2 test split.

---

## ⚙️ Experimental Setup

### Models

| Model      | Layers                                               | Notes                           |
| :--------- | :--------------------------------------------------- | :------------------------------ |
| **CNN**    | Conv1D(128, 5) → GlobalMaxPool → Dense(20) → Softmax | Lightweight sentence classifier |
| **BiLSTM** | BiLSTM(64 → 32) + Dropout(0.5) → Dense(20) → Softmax | Sequential context model        |

**Embeddings:** GloVe 6B 100-dimensional  
**Loss:** Categorical Cross-Entropy  **Optimizer:** Adam  
**Max tokens:** 50  **Validation split:** 0.1  **Early Stopping:** 3 epochs  
**Hardware:** Google Colab A100 GPU

---

### Augmentation Methods

| Method              | Symbol | Range (α)  | Example                              |
| :------------------ | :----: | :--------: | :----------------------------------- |
| Synonym Replacement |   SR   | 0.05 – 0.5 | “movie was great” → “film was great” |
| Random Insertion    |   RI   | 0.05 – 0.5 | insert synonyms at random positions  |
| Random Swap         |   RS   | 0.05 – 0.5 | swap two tokens                      |
| Random Deletion     |   RD   | 0.05 – 0.5 | drop tokens with prob. α             |

Each method produces `train_<method>_<alpha>.txt` inside its corresponding size folder.

---

## 📊 Metrics

Both **train** and **test** evaluations report:

- Accuracy   - Precision   - Recall   - F1

## 🚀 Steps to Run

1. **Clone & Setup**

   ```bash
   git clone https://github.com/your-repo/ML_EDA_FALL2025
   cd ML_EDA_FALL2025
   pip install -r requirements.txt
   ```

2. **Download Data**

   ```bash
   # Download SST-2 dataset
   python download_data.py

   # Download GloVe embeddings
   wget https://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   ```

3. **Generate Augmentations**

   ```bash
   # Run augmentation notebook
   jupyter notebook 01_augment_ml.ipynb
   ```

4. **Train Models**

   ```bash
   # Train on different data splits
   python 02_train.py --model cnn --split tiny
   python 02_train.py --model bilstm --split full
   ```

---

- AUROC (ROC AUC)
- AUC-PR
- Confusion Matrix
- ROC / PR Curves

All metrics are logged in per-run `.json` files inside `/models`.
