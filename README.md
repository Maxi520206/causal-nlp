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

## Project Structure

```text
causal-nlp
├── src/                # core model and training code
├── notebooks/          # exploratory analysis and drafts
├── artifacts/          # saved outputs and figures
├── experiments/
│   ├── baseline/       # initial IMDB baseline
│   ├── spurious/       # spurious correlation injection
│   ├── cda/            # counterfactual data augmentation
│   ├── adjustment/     # causal adjustment methods
│   └── probing/        # representation probing
├── requirements.txt
└── README.md
