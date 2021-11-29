import argparse

from transformers import pipeline


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sentence", type=str, default=None)

    args = parser.parse_args()

    return args


def init():
    model = pipeline("text-classification")

    return model


if __name__ == "__main__":
    args = get_args()

    model = init()

    result = model(args.sentence)

    print(result["label"], result["score"])
    