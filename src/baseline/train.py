# src/train.py
import argparse
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.baseline.data import load_imdb_splits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_features", type=int, default=20000)
    p.add_argument("--ngram_max", type=int, default=1)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--out_dir", type=str, default="artifacts/baseline")
    return p.parse_args()


def main():
    args = parse_args()

    print("TRAIN SCRIPT STARTED", flush=True)

    print("Loading data...", flush=True)
    X_train, y_train, X_val, y_val, _, _ = load_imdb_splits(
        test_size=args.test_size,
        seed=args.seed
    )
    print("Data loaded:", len(X_train), len(X_val), flush=True)

    print("Vectorizing...", flush=True)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    print("Vectorized:", X_train_vec.shape, X_val_vec.shape, flush=True)

    print("Training LR...", flush=True)
    clf = LogisticRegression(C=args.C, max_iter=args.max_iter)
    clf.fit(X_train_vec, y_train)
    print("Training done.", flush=True)

    print("Evaluating on VAL...", flush=True)
    val_pred = clf.predict(X_val_vec)
    val_acc = accuracy_score(y_val, val_pred)
    val_cm = confusion_matrix(y_val, val_pred)

    print(f"[VAL] accuracy: {val_acc:.4f}")
    print("[VAL] confusion matrix:\n", val_cm)
    print("[VAL] classification report:\n",
          classification_report(y_val, val_pred, digits=4))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, out_dir / "tfidf.joblib")
    joblib.dump(clf, out_dir / "logreg.joblib")

    print(f"Saved: {(out_dir / 'tfidf.joblib').as_posix()}")
    print(f"Saved: {(out_dir / 'logreg.joblib').as_posix()}")


if __name__ == "__main__":
    main()
