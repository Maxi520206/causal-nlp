from src.baseline.train import train
from src.baseline.eval import evaluate


def main():

    model, vectorizer = train()

    evaluate(model, vectorizer)


if __name__ == "__main__":
    main()
