# src/eval.py
import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.data import load_imdb_splits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact_dir", type=str, default="artifacts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)  # 为了保持和 train 一致
    p.add_argument("--save_fig", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print("EVAL SCRIPT STARTED", flush=True)

    artifact_dir = Path(args.artifact_dir)
    tfidf_path = artifact_dir / "tfidf.joblib"
    model_path = artifact_dir / "logreg.joblib"

    print("Looking for artifacts:", tfidf_path, model_path, flush=True)
    if not tfidf_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Missing artifacts. Run `python -m src.train` first to generate "
            "`artifacts/tfidf.joblib` and `artifacts/logreg.joblib`."
        )

    vectorizer = joblib.load(tfidf_path)
    clf = joblib.load(model_path)
    print("Artifacts loaded.", flush=True)

    print("Loading test data...", flush=True)
    _, _, _, _, X_test, y_test = load_imdb_splits(test_size=args.test_size, seed=args.seed)

    print("Vectorizing test...", flush=True)
    X_test_vec = vectorizer.transform(X_test)

    print("Predicting...", flush=True)
    test_pred = clf.predict(X_test_vec)

    test_acc = accuracy_score(y_test, test_pred)
    test_cm = confusion_matrix(y_test, test_pred)

    print(f"[TEST] accuracy: {test_acc:.4f}")
    print("[TEST] confusion matrix:\n", test_cm)
    print("[TEST] classification report:\n", classification_report(y_test, test_pred, digits=4))

    disp = ConfusionMatrixDisplay.from_predictions(y_test, test_pred)
    plt.title("IMDB - TFIDF + Logistic Regression (Test)")

    if args.save_fig:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        out_path = artifact_dir / "confusion_matrix_test.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {out_path.as_posix()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
