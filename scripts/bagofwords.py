from tokenize_Stijn import Tokens
from nlp_jarno import Bagofwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sys
from math import log


def extract_abstracts(filepath):
    output_dict = {}
    with open(filepath, "r") as train:
        for line in train.read().split("\n"):
            header, abstract = line.split("]:")
            output_dict[header[1:]] = abstract
    return output_dict


def main():
    # inputs = sys.argv[1:]
    # print(inputs)
    # cancer_abstracts = extract_abstracts(inputs[0])
    # other_abstracts = extract_abstracts(inputs[1])
    # y = [i < len(cancer_abstracts) for i in range(len(cancer_abstracts) + len(other_abstracts))]
    # print(trainset.keys())
    # tests
    tokens = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h", 9: "i", 10: "j", 11: "k", 12: "l",
              13: "m", 14: "n", 15: "o", 16: "p", 17: "q", 18: "r", 19: "s", 20: "t", 21: "u", 22: "v", 23: "w",
              24: "x", 25: "y", 26: "z", 27: " ", 28: "."}
    X = ["hello.", "help", "hooray", "beckit", "dammaret", "wisconsin"]
    y = [0, 0, 1, 1, 1, 0]
    bag = Bagofwords(tokens)
    bag.fit(X, y)
    print(bag.predict(["happy"]))


if __name__ == "__main__":
    main()

