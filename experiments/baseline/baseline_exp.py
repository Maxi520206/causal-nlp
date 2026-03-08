import yaml

from src.baseline.train import train
from src.baseline.eval import evaluate


def main():

    with open("config/baseline_v1.yaml", "r") as f:
        config = yaml.safe_load(f)

    model, vectorizer = train(config)

    evaluate(model, vectorizer, config)


if __name__ == "__main__":
    main()
