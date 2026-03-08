# Baseline v1 Experiment Log

## Objective

The goal of this experiment was to establish a **reproducible baseline model** for the Causal NLP project.

Before studying causal mechanisms such as shortcut learning, spurious correlations, and distribution shifts, it is necessary to define a **stable reference model**. This baseline will serve as the comparison point for all subsequent experiments in the research pipeline.

The experiment focuses on sentiment classification using the **IMDB dataset**, implementing a classical NLP pipeline as the starting point of the project.

---

## Baseline Model

The baseline model follows a traditional text classification setup.

### Text Representation
- TF-IDF vectorization

### Classifier
- Logistic Regression

This combination is widely used as a strong and interpretable baseline in NLP tasks.

The training pipeline consists of:

1. Loading the IMDB dataset
2. Converting text into TF-IDF feature vectors
3. Training a logistic regression classifier
4. Evaluating performance on a validation set
5. Saving the trained artifacts for reproducibility

Saved artifacts include:

```
tfidf.joblib
logreg.joblib
```

These artifacts are stored under:

```
artifacts/baseline/
```

---

## Experimental Results

Validation performance:

**Accuracy ≈ 0.8876**

Classification report:

| Class | Precision | Recall | F1 |
|------|-----------|--------|----|
| Negative | 0.9004 | 0.8716 | 0.8858 |
| Positive | 0.8756 | 0.9036 | 0.8894 |

This result is consistent with typical TF-IDF + Logistic Regression baselines reported for the IMDB dataset.

---

## Reproducibility Protocol

To ensure reliable experimental results, the baseline was trained using **multiple random seeds**.

Three independent runs were executed:

```
seed = 1
seed = 2
seed = 3
```

Each run saves its artifacts in a separate directory:

```
artifacts/baseline/seed1
artifacts/baseline/seed2
artifacts/baseline/seed3
```

This setup ensures that results are not dependent on a single random initialization and follows standard machine learning research practices.

---

## Role in the Research Pipeline

This baseline experiment corresponds to **Step 1 of the overall research pipeline**:

1. Baseline sentiment classifier  
2. Spurious correlation injection  
3. Counterfactual data augmentation  
4. Causal adjustment methods  
5. Representation probing  

The baseline establishes the model’s behavior under standard training conditions.

Future experiments will analyze how the model behaves under controlled perturbations and causal interventions.

---

## Next Steps

With the baseline established, the next step will be to introduce **controlled spurious correlations** into the dataset.

This will allow us to study:

- whether the model relies on shortcut features
- how performance changes under distribution shifts
- how causal interventions affect learned representations

These experiments will form the foundation for the causal analysis of NLP models.
