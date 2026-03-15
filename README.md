# From Statistical Features to Contextual Embeddings: A Multi-Paradigm Evaluation of Emotion Classification Models

**Author:** Gregorius Reynaldi Pratama  
**Institution:** Department of Computer Science, Harbin Institute of Technology Shenzhen  

---

## Overview

This repository contains the full implementation for the paper *"From Statistical Features to Contextual Embeddings: A Multi-Paradigm Evaluation of Emotion Classification Models"*. The study systematically evaluates 14 model variants across three representational paradigms — classical machine learning, pre-trained word embedding-based deep learning, and transformer fine-tuning — on a six-class emotion classification task.

The six emotion categories are: **anger, fear, joy, love, sadness, and surprise**.

---

## Repository Structure

```
.
├── datasets/
│   ├── material/
│   │   ├── training.xlsx          # Training data
│   │   ├── test.xlsx              # Test data
│   │   └── readyDataset.pkl       # Preprocessed dataset (generated)
│   ├── processed/                 # Processed embedding arrays (generated)
│   └── GLOVE/                     # GloVe vocab files (download required)
│       ├── gloveVocab.txt         # Place downloaded GloVe file here
│       └── word2vecVocab.txt      # Auto-generated from GloVe conversion
│   └── FAST_TEXT/                 # FastText vocab files (download required)
│       └── fastTextVocab.vec      # Place downloaded FastText file here
├── models/
│   ├── transformerModels/         # Fine-tuned transformer checkpoints (generated)
│   ├── gloveModel.pkl             # Serialized GloVe model (generated)
│   └── fastTextModel.pkl          # Serialized FastText model (generated)
├── documentationPaper/
│   └── PAPER.pdf                  # Published paper
├── src/
│   ├── exploratory_data_analysis.ipynb
│   ├── ml_baseline.ipynb
│   ├── gloveAndFastText.ipynb
│   ├── transformers_approach.ipynb
│   └── RESULT.md
├── requirements.txt
└── README.md
```

> **Note:** Large files including model weights, embedding vocabularies, and serialized `.pkl` files are excluded from this repository. Follow the setup instructions below to prepare all required assets before running the notebooks.

---

## Requirements

### Python Version

Python 3.9 or higher is recommended.

### Install Dependencies

Using `requirements.txt` (recommended):
```bash
pip install -r requirements.txt
```

If you are using a CUDA-enabled GPU (CUDA 11.8), use:
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

Then open `requirements.txt` and uncomment the three GPU PyTorch lines, and comment out the three CPU lines.

If you prefer to install manually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
pip install tensorflow keras
pip install torch torchvision torchaudio
pip install transformers datasets torchinfo
pip install gensim nltk openpyxl
```

For GPU support, replace the PyTorch line above with:
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## Data Preparation

### 1. Emotion Dataset

The dataset used in this study is the **Emotion Dataset** by Parulpandey, available on Kaggle.

**Download:**
- Go to: https://www.kaggle.com/datasets/parulpandey/emotion-dataset
- Download the dataset files: `training.csv` and `test.csv` (or their `.xlsx` equivalents)

**Placement:**

Convert or save the files as `.xlsx` format and place them in the following directory:

```
datasets/material/training.xlsx
datasets/material/test.xlsx
```

The dataset contains two columns: `text` and `label`. Labels are string values: anger, fear, joy, love, sadness, surprise.

---

### 2. GloVe Pre-Trained Vectors

This study uses the **GloVe 6B 300-dimensional** vectors trained on Wikipedia 2014 and Gigaword 5 (6B tokens, 400K vocabulary, uncased).

**Download:**
- Go to: https://nlp.stanford.edu/projects/glove/
- Download: `glove.6B.zip` (822 MB)

**Setup:**

After downloading, extract the archive and locate the file `glove.6B.300d.txt`. Rename it and place it as follows:

```
datasets/GLOVE/gloveVocab.txt
```

The `word2vecVocab.txt` file in the same directory will be auto-generated when you run `gloveAndFastText.ipynb` via the `glove2word2vec` conversion step. You do not need to create it manually.

---

### 3. FastText Pre-Trained Vectors

This study uses the **wiki-news-300d-1M** FastText vectors trained on Wikipedia 2017, UMBC WebBase corpus, and statmt.org news dataset (16B tokens, 1 million word vectors).

**Download:**
- Go to: https://fasttext.cc/docs/en/english-vectors.html
- Download: `wiki-news-300d-1M.vec.zip`

**Setup:**

After downloading, extract the archive to get `wiki-news-300d-1M.vec`. Rename it and place it as follows:

```
datasets/FAST_TEXT/fastTextVocab.vec
```

---

## Running the Notebooks

The notebooks are designed to be run sequentially. Each notebook depends on outputs generated by the previous one.

### Step 1 — Exploratory Data Analysis

```
src/exploratory_data_analysis.ipynb
```

This notebook loads the raw dataset, performs label distribution analysis, calculates token length statistics, applies the preprocessing pipeline (special character removal, whitespace normalization, lowercasing), and saves the cleaned dataset as:

```
datasets/material/readyDataset.pkl
```

Run this notebook first before anything else.

---

### Step 2 — Baseline Machine Learning Models

```
src/ml_baseline.ipynb
```

This notebook loads `readyDataset.pkl` and trains the following classifiers across three vectorizer configurations (CountVectorizer, TF-IDF, TF-IDF with bigrams):

- Naive Bayes
- Logistic Regression
- LinearSVC
- XGBoost
- MLP (Multi-Layer Perceptron)

No additional files are required beyond `readyDataset.pkl`.

---

### Step 3 — GloVe and FastText Embedding Models

```
src/gloveAndFastText.ipynb
```

This notebook requires both GloVe and FastText vocabulary files to be in place before running. It performs the following:

1. Converts GloVe format to Word2Vec format and saves to `datasets/GLOVE/word2vecVocab.txt`
2. Loads both embedding models and serializes them to `models/gloveModel.pkl` and `models/fastTextModel.pkl`
3. Builds sentence embedding matrices and saves processed arrays to `datasets/processed/`
4. Trains and evaluates the following architectures:
   - Bidirectional GRU with combined GloVe + FastText (600-dim input)
   - Bidirectional LSTM with combined GloVe + FastText (600-dim input)
   - Bidirectional GRU with GloVe only (300-dim input)
   - Bidirectional GRU with FastText only (300-dim input)

**Before running**, ensure the following files exist:

```
datasets/GLOVE/gloveVocab.txt
datasets/FAST_TEXT/fastTextVocab.vec
```

---

### Step 4 — Transformer Fine-Tuning

```
src/transformers_approach.ipynb
```

This notebook fine-tunes three transformer models for the six-class emotion classification task:

- `distilbert-base-uncased` (DistilBERT)
- `bert-base-uncased` (BERT)
- `roberta-base` (RoBERTa)

Model weights are downloaded automatically from Hugging Face on first run. A CUDA-enabled GPU is strongly recommended. Fine-tuned checkpoints are saved to:

```
models/transformerModels/distilBertModel/
models/transformerModels/bertModel/
models/transformerModels/RoBertaModel/
```

**Training configuration used in the study:**

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Batch size (train) | 16 |
| Batch size (eval) | 8 |
| Max epochs | 10 |
| Mixed precision | fp16 |
| Best checkpoint metric | Weighted F1-score |

---

## Results Summary

All models were evaluated on a fixed 4,000-sample test set. The table below summarizes the top-performing model from each paradigm.

| Rank | Model | Paradigm | Accuracy | F1-Score |
|------|-------|----------|----------|----------|
| 1 | BERT | Transformer | 0.936 | 0.937 |
| 1 | RoBERTa | Transformer | 0.936 | 0.937 |
| 3 | BiGRU (GloVe + FastText) | Embedding-based | 0.932 | 0.932 |
| 3 | DistilBERT | Transformer | 0.932 | 0.932 |
| 5 | BiGRU (GloVe only) | Embedding-based | 0.929 | 0.929 |
| 6 | BiGRU (FastText only) | Embedding-based | 0.925 | 0.927 |
| 7 | BiLSTM (GloVe + FastText) | Embedding-based | 0.920 | 0.921 |
| 8 | LinearSVC (TF-IDF) | Classical ML | 0.893 | 0.892 |
| 9 | Logistic Regression (CountVec) | Classical ML | 0.885 | 0.884 |
| 10 | XGBoost (CountVec) | Classical ML | 0.884 | 0.885 |

The full results including per-class metrics and training times are available in `src/RESULT.md`.

---

## Key Findings

The transition from sparse statistical features to pre-trained GloVe embeddings accounts for the largest single performance gain in this study (+3.6 F1 points), representing the most fundamental shift in representational capacity. Full transformer fine-tuning with BERT and RoBERTa yields only a marginal additional improvement (+0.8 points) at substantially higher computational cost.

For resource-constrained deployment, the Bidirectional GRU with GloVe-only embeddings (92.9% F1, 1.74 MB, no GPU required for inference) represents a near-optimal balance between performance and efficiency. The full analysis and discussion are available in the published paper under `references/PAPER.pdf`.

---

## License

This repository is released for academic and research purposes. The pre-trained GloVe and FastText vectors are subject to their respective licenses from Stanford NLP and Facebook Research. The Emotion Dataset is subject to its license on Kaggle.