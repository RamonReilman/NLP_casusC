from tokenize_Stijn import Tokens
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sys
from math import log


class Bagofwords:
    def __init__(self, tokens, encoding_type="multi_hot"):
        if encoding_type not in ["multi_hot", "frequency", "tf_idf"]:
            raise ValueError("encoding_type unknown: " + encoding_type)
        self.model = GaussianNB()
        self.tokens = tokens
        self.tot_counts = {i: 0 for i in tokens.keys()}
        self.encoding_type = encoding_type

    def create_bag(self, X):
        X_tokens = []
        X_encoded = []

        # convert strings to tokens
        tokenise = Tokens(self.tokens)
        for x_i in X:
            X_tokens.append(tokenise.string_to_tokens(x_i))

        # count total amount of tokens if total count is 0 (hasn't been counted yet)
        if sum(self.tot_counts.values()) == 0:
            for x_i_token in X_tokens:
                for token in x_i_token:
                    self.tot_counts[token] = self.tot_counts[token] + 1

        # use encoding based on given encoding type
        for x_i_token in X_tokens:
            local_counts = {i: 0 for i in self.tot_counts.keys()}
            for token in x_i_token:
                local_counts[token] = local_counts[token] + 1
            if self.encoding_type == "multi_hot":
                X_encoded.append(self.multi_hot(local_counts))
            elif self.encoding_type == "frequency":
                X_encoded.append(self.frequency(local_counts))
            elif self.encoding_type == "tf_idf":
                X_encoded.append(self.tf_idf(local_counts))
        return X_encoded

    @staticmethod
    def multi_hot(token_counts):
        return [bool(count) for count in token_counts.values()]

    @staticmethod
    def frequency(token_counts):
        tot_count = sum(token_counts.values())
        return [count / tot_count for count in token_counts.values()]

    def tf_idf(self, local_token_counts):
        full_tot_count = sum(self.tot_counts.values())
        local_tot_count = sum(local_token_counts.values())
        return [(local_count / local_tot_count) * log(full_count / full_tot_count)
                for full_count, local_count in zip(self.tot_counts.values(), local_token_counts.values())]

    def fit(self, X, y):
        self.model.fit(self.create_bag(X), y)

    def predict(self, X):
        return self.model.predict(self.create_bag(X))

    def __str__(self):
        return f"model: {self.model}, provided_tokens: {self.tokens}, counted tokens in trainset: {self.tot_counts}"


def extract_abstracts(filepath):
    output_dict = {}
    with open(filepath, "r") as train:
        for line in train.read().split("\n"):
            header, abstract = line.split("]:")
            output_dict[header[1:]] = abstract
    return output_dict


def main():
    inputs = sys.argv[1:]
    print(inputs)
    cancer_abstracts = extract_abstracts(inputs[0])
    other_abstracts = extract_abstracts(inputs[1])
    y = [i < len(cancer_abstracts) for i in range(len(cancer_abstracts) + len(other_abstracts))]
    # print(trainset.keys())
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

