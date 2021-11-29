import argparse

from transformers import pipeline


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--context", type=str, default="""
    The United States of America (U.S.A. or USA), commonly known as the United States (U.S. or US) or America, is a country primarily located in North America. It consists of 50 states, a federal district, five major unincorporated territories, 326 Indian reservations, and some minor possessions.[j] At 3.8 million square miles (9.8 million square kilometers), it is the world's third- or fourth-largest country by total area.[e] The United States shares significant land borders with Canada to the north and Mexico to the south as well as limited maritime borders with the Bahamas, Cuba, and Russia.[20] With a population of more than 331 million people, it is the third most populous country in the world. The national capital is Washington, D.C., and the most populous city is New York City.
    """)
    parser.add_argument("--question", type=str, default="How wide is the US?")

    args = parser.parse_args()

    return args


def init():
    model = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

    return model


if __name__ == "__main__":
    args = get_args()

    model = init()
    result = model(question=args.question, context=args.context)

    print(result["answer"], result["score"])
