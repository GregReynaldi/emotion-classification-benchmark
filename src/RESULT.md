# Performance Summary: Text Classification Models

## Overview

This document presents a comprehensive performance analysis of various text classification models implemented for the project. The models are categorized into three main approaches:

1. **Baseline Machine Learning Models**
2. **GloVe and FastText Embedding-based Models**
3. **Transformer-based Models**

All models were evaluated on the same test dataset of 4,000 samples, ensuring a fair comparison across different approaches.

## 1. Baseline Machine Learning Models

### 1.1 Naive Bayes
| Vectorizer | Accuracy | Precision | Recall | F1-Score | Train Time (s) |
|------------|----------|-----------|--------|----------|----------------|
| Count Vectorizer | 0.835 | 0.832 | 0.835 | 0.830 | 2.64 |
| TF-IDF | 0.729 | 0.787 | 0.729 | 0.682 | 0.12 |
| TF-IDF + NGram | 0.685 | 0.769 | 0.685 | 0.624 | 0.58 |

### 1.2 Logistic Regression
| Vectorizer | Accuracy | Precision | Recall | F1-Score | Train Time (s) |
|------------|----------|-----------|--------|----------|----------------|
| Count Vectorizer | 0.885 | 0.884 | 0.885 | 0.884 | 3.19 |
| TF-IDF | 0.866 | 0.868 | 0.866 | 0.861 | 3.23 |
| TF-IDF + NGram | 0.834 | 0.843 | 0.834 | 0.825 | 8.05 |

### 1.3 Support Vector Machine (SVC)
| Vectorizer | Accuracy | Precision | Recall | F1-Score | Train Time (s) |
|------------|----------|-----------|--------|----------|----------------|
| Count Vectorizer | 0.880 | 0.880 | 0.880 | 0.880 | 1.30 |
| TF-IDF | 0.893 | 0.892 | 0.893 | 0.892 | 0.55 |
| TF-IDF + NGram | 0.887 | 0.886 | 0.887 | 0.886 | 1.49 |

### 1.4 XGBoost
| Vectorizer | Accuracy | Precision | Recall | F1-Score | Train Time (s) |
|------------|----------|-----------|--------|----------|----------------|
| Count Vectorizer | 0.884 | 0.887 | 0.884 | 0.885 | 3.53 |
| TF-IDF | 0.872 | 0.874 | 0.872 | 0.873 | 12.31 |
| TF-IDF + NGram | 0.876 | 0.878 | 0.876 | 0.877 | 37.33 |

### 1.5 MLP (Multi-Layer Perceptron)
| Metric | Value |
|--------|-------|
| Accuracy | 0.828 |
| Precision | 0.836 |
| Recall | 0.828 |
| F1-Score | 0.830 |
| Train Time (s) | 86.57 |

## 2. GloVe and FastText Embedding-based Models

### 2.1 Bidirectional GRU (Combined GloVe + FastText)
| Metric | Value |
|--------|-------|
| Accuracy | 0.932 |
| Precision | 0.933 |
| Recall | 0.932 |
| F1-Score | 0.932 |
| Train Time (s) | 1035.09 |
| Model Size | 2.62 MB |

### 2.2 Bidirectional GRU (GloVe Only)
| Metric | Value |
|--------|-------|
| Accuracy | 0.929 |
| Precision | 0.930 |
| Recall | 0.929 |
| F1-Score | 0.929 |
| Train Time (s) | 881.39 |
| Model Size | 1.74 MB |

### 2.3 Bidirectional GRU (FastText Only)
| Metric | Value |
|--------|-------|
| Accuracy | 0.925 |
| Precision | 0.931 |
| Recall | 0.925 |
| F1-Score | 0.927 |
| Train Time (s) | 1002.61 |
| Model Size | 1.74 MB |

### 2.4 Bidirectional LSTM (Combined GloVe + FastText)
| Metric | Value |
|--------|-------|
| Accuracy | 0.920 |
| Precision | 0.923 |
| Recall | 0.920 |
| F1-Score | 0.921 |
| Train Time (s) | 1869.46 |
| Model Size | 3.48 MB |

## 3. Transformer-based Models

### 3.1 DistilBERT
| Metric | Value |
|--------|-------|
| Accuracy | 0.932 |
| Precision | 0.934 |
| Recall | 0.932 |
| F1-Score | 0.932 |
| Model Size | 66.96M parameters |

### 3.2 BERT
| Metric | Value |
|--------|-------|
| Accuracy | 0.936 |
| Precision | 0.938 |
| Recall | 0.936 |
| F1-Score | 0.937 |
| Model Size | 109.49M parameters |

### 3.3 RoBERTa
| Metric | Value |
|--------|-------|
| Accuracy | 0.936 |
| Precision | 0.940 |
| Recall | 0.936 |
| F1-Score | 0.937 |
| Model Size | 124.65M parameters |

## Performance Comparison

### Top Performing Models by Accuracy

| Rank | Model | Accuracy | F1-Score |
|------|-------|----------|----------|
| 1 | BERT | 0.936 | 0.937 |
| 1 | RoBERTa | 0.936 | 0.937 |
| 3 | Bidirectional GRU (GloVe+FastText) | 0.932 | 0.932 |
| 3 | DistilBERT | 0.932 | 0.932 |
| 5 | Bidirectional GRU (GloVe Only) | 0.929 | 0.929 |
| 6 | Bidirectional GRU (FastText Only) | 0.925 | 0.927 |
| 7 | Bidirectional LSTM (GloVe+FastText) | 0.920 | 0.921 |
| 8 | SVC (TF-IDF) | 0.893 | 0.892 |
| 9 | Logistic Regression (Count Vectorizer) | 0.885 | 0.884 |
| 10 | XGBoost (Count Vectorizer) | 0.884 | 0.885 |

### Computational Efficiency

| Model Type | Average Train Time (s) | Model Size |
|------------|------------------------|------------|
| Baseline ML | 7.79 | Small |
| GloVe+FastText | 1247.14 | Medium |
| Transformers | N/A (not explicitly measured) | Large |

## Key Insights

1. **Transformer-based models** (BERT, RoBERTa, DistilBERT) achieve the highest performance with accuracy scores around 93.6% and F1-scores around 93.7%.

2. **Embedding-based models** show varying performance based on the embedding type:
   - Bidirectional GRU with combined GloVe+FastText embeddings: 93.2% accuracy
   - Bidirectional GRU with only GloVe embeddings: 92.9% accuracy
   - Bidirectional GRU with only FastText embeddings: 92.5% accuracy
   - Bidirectional LSTM with combined GloVe+FastText embeddings: 92.0% accuracy

3. **Baseline machine learning models** show varying performance, with SVC (TF-IDF) being the best performer among them with an accuracy of 89.3% and F1-score of 89.2% while maintaining fast training times.

4. **Training time** increases significantly with model complexity, with transformer models and embedding-based models requiring substantially more computational resources compared to baseline models. Among embedding-based models, Bidirectional GRU with GloVe only is the fastest to train (881.39s), followed by Bidirectional GRU with FastText only (1002.61s) and Bidirectional GRU with combined embeddings (1035.09s).

5. **Model size** follows a similar pattern, with transformer models being the largest (tens of millions of parameters), followed by embedding-based models (1.74MB to 3.48MB), and baseline models being the smallest. The Bidirectional GRU models with single embeddings (GloVe or FastText) have smaller model sizes (1.74MB) compared to those with combined embeddings (2.62MB).

6. **Embedding type impact**: Using combined GloVe+FastText embeddings provides a slight performance improvement over using either embedding alone, suggesting that the combination captures complementary information from both embedding types.

## Conclusion

The transformer-based models (BERT and RoBERTa) demonstrate the best overall performance for text classification in this project, achieving the highest accuracy and F1-score. However, they come with increased computational requirements and model size.

For scenarios with limited computational resources, the Bidirectional GRU models offer a good balance between performance and resource usage:
- Bidirectional GRU with combined GloVe and FastText embeddings: 93.2% accuracy, 1035.09s training time, 2.62MB model size
- Bidirectional GRU with only GloVe embeddings: 92.9% accuracy, 881.39s training time, 1.74MB model size
- Bidirectional GRU with only FastText embeddings: 92.5% accuracy, 1002.61s training time, 1.74MB model size

The Bidirectional GRU with only GloVe embeddings stands out as the most efficient option among the embedding-based models, offering nearly the same performance as the combined embeddings version but with faster training times and a smaller model size.

Among the baseline models, SVC with TF-IDF vectorization provides the best performance with relatively fast training times, making it a suitable choice for deployment in resource-constrained environments.

Overall, the choice of model should be based on a balance between performance requirements, available computational resources, and deployment constraints. For applications where performance is paramount, transformer models are the best choice. For applications where efficiency is a concern, the Bidirectional GRU with GloVe only embeddings offers an excellent balance of performance and resource usage.
