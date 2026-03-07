# IMDB Sentiment Classification – NLP Baseline 20260118

This repository contains my **first end-to-end NLP project**, built as part of my learning process in Natural Language Processing.

The goal of this project is not only to achieve reasonable performance, but more importantly to **understand the full pipeline of a classic text classification task**, from text representation to model evaluation.

---

## Project Overview

- **Task**: Binary sentiment classification (Positive / Negative)
- **Dataset**: IMDB Movie Reviews
- **Model**: TF-IDF + Logistic Regression
- **Purpose**: Learning-oriented baseline

This project serves as a **reproducible and interpretable baseline**, as well as a personal learning log documenting my progress in NLP.

---

## Learning Objectives

Through this project, I aimed to:

- Understand how raw text is converted into numerical features
- Build a complete NLP classification pipeline from scratch
- Learn how to evaluate classification models beyond accuracy
- Develop intuition about the strengths and limitations of bag-of-words models

---

## Methodology

### Text Representation

- **TF-IDF (Term Frequency–Inverse Document Frequency)**
- Bag-of-words style representation
- Captures word importance while remaining simple and interpretable

### Classifier

- **Logistic Regression**
- Linear model commonly used as a strong baseline in text classification
- Efficient to train and easy to analyze

This combination is a standard baseline in sentiment analysis and provides a solid reference point for more advanced models.

---

## Experimental Setup

- **Dataset**: IMDB Movie Reviews
- **Test Set Size**: 25,000 samples
- **Task Type**: Binary classification

The model is evaluated on the test set using multiple metrics to obtain a balanced view of performance.

---

## Results

### Test Accuracy
- Accuracy: 0.8768
### Confusion Matrix (Test)

|                | Predicted Negative | Predicted Positive |
|----------------|-------------------|-------------------|
| True Negative  | 10,936            | 1,564             |
| True Positive  | 1,517             | 10,983            |

The confusion matrix shows that the model performs similarly on both classes, with comparable numbers of false positives and false negatives. This suggests no strong class bias.

### Classification Report

- **Negative class (0)**
  - Precision: 0.878
  - Recall: 0.875
  - F1-score: 0.877

- **Positive class (1)**
  - Precision: 0.875
  - Recall: 0.879
  - F1-score: 0.877

---

## Analysis and Observations

From the results, several observations can be made:

- TF-IDF features combined with a linear model already capture a significant amount of sentiment information
- Most misclassifications occur in reviews with:
  - Mixed sentiment
  - Ambiguous or nuanced expressions
- This behavior is expected, as bag-of-words models lack contextual and semantic understanding

Overall, this baseline achieves **strong and balanced performance** for a simple linear model.

---

## What I Learned

Through this project, I gained practical experience with:

- The standard NLP pipeline for text classification
- Strengths and limitations of TF-IDF representations
- Interpreting confusion matrices and classification reports
- Understanding why linear models struggle with complex semantics

This project helped bridge the gap between theoretical knowledge and hands-on implementation.

---

## Next Steps

Planned improvements and future experiments include:

- Hyperparameter tuning for TF-IDF (e.g., n-grams, min_df)
- Trying alternative traditional models (e.g., SVM)
- Exploring neural approaches:
  - CNN / LSTM-based models
  - Pretrained language models (e.g., BERT)
- Conducting more systematic error analysis

---




