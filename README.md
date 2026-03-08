# Causal NLP Project

This repository studies shortcut learning and causal mechanisms in NLP.

Modern NLP models often achieve high accuracy by exploiting
spurious correlations rather than learning stable causal mechanisms.

This project builds an experimental pipeline to investigate this problem
using the IMDB sentiment classification task.

Research pipeline:

1. Baseline sentiment classifier
2. Spurious correlation injection
3. Counterfactual data augmentation
4. Causal adjustment methods
5. Representation probing

### Baseline v1 (implemented)

The first step of the pipeline is a reproducible IMDB sentiment classification baseline using **TF-IDF + Logistic Regression**.

Reproducibility protocol:

- fixed dataset split
- 3 random seeds (1,2,3)
- model artifacts saved for each run

Example run:

python -m src.baseline.train --seed 1

## Project Structure

```text
causal-nlp
├── config/
├── docs/
│   └── research_logs/
├── experiments/
│   ├── baseline/
│   ├── spurious/
│   ├── cda/
│   ├── adjustment/
│   └── probing/
├── notebooks/
├── src/
│   └── baseline/
├── requirements.txt
├── .gitignore
└── README.md
