import math
import numpy as np
from tokenize_Ramon import read_file
import pathlib
from sklearn.neural_network import MLPClassifier
import argparse

def to_path(path):
    pathed = pathlib.Path(path)
    if pathed.exists():
        return pathed
    raise FileNotFoundError(path)


def setup_parser():
    parser = argparse.ArgumentParser(
        prog="Embed",
        description="For all your word-embedding needs",
        epilog="help"
    )
    parser.add_argument("-t","--tok", type=to_path, required=True,
                        help="Path to a tok file")
    parser.add_argument("-n", "--ngram", default=2,
                        help = "Window of the 2N+1 gram", type=int)
    parser.add_argument("-o", "--output", default="output.emb",
                        help= "Path to write .emb file to. (default = ./output.emb)")
    parser.add_argument("-nn", "--n_hidden", default=5, type=int,
                        help = "Amount of neurons the hidden layer should contain. (default = 5)")
    return parser

def create_ngrams(tokens, n):
    n_gram = []
    for sentence in tokens:
        temp_gram = []
        for idx, token in enumerate(sentence):
            token_gram = []
            if idx - n >= 0 and idx + n < len(sentence):
                for i in range(-n, n+1):
                    token_gram.append(sentence[idx+i])
                temp_gram.append(token_gram)
        n_gram.append(temp_gram)

    return n_gram

def multi_hot_encoding(vocab):
    generate_token_dict(vocab)

    multi_hot_ = np.zeros((len(vocab),len(vocab)))
    for i in range(len(vocab)):
        multi_hot_[i][i] = 1

    multi_hot_dict = {}
    for i in range(len(vocab)):
        multi_hot_dict[vocab[i]] = multi_hot_[i]
    return multi_hot_dict


def generate_token_dict(vocab):
    counter = 0
    token_dict = {}
    for word in vocab:
        token_dict[word] = counter
        counter += 1
    return token_dict


def create_vocab(tokens):
    vocab = list(set([token for sentence in tokens for token in sentence]))
    vocab.sort()
    return vocab


def generate_x_y(multi_hot_dict, n_grams):
    X = []
    Y = []
    for sentence in n_grams:
        for n_gram in sentence:
            temp_x = np.zeros(len(multi_hot_dict))
            n = len(n_gram)
            middle = math.floor(n/2)
            Y.append(multi_hot_dict.get(n_gram[middle]))
            for idx in range(0 , n):
                if idx != middle:
                    value = multi_hot_dict.get(n_gram[idx])
                    temp_x += value
            X.append(np.clip(temp_x, 0, 1))
    return X,Y

def create_model(n_neurons):
    model = MLPClassifier(hidden_layer_sizes=(n_neurons,))
    return model


def train_model(model, X, Y):
    model.fit(X,Y)
    return model

def get_weights(model):
    return model.coefs_[1].T


def get_embedding(vocab, weights):
    embedding = {}
    token_dict = generate_token_dict(vocab)
    for token in vocab:
        embedding[token] = weights[token_dict[token]]

    return embedding

def write_emb(embedding, output=None):
    with open(output, 'w') as f:
        for embed, value in embedding.items():
            s = f"{embed}"
            for val in value:
                s += f" {val}"
            s += "\n"
            f.write(s)

def main():
    args = setup_parser().parse_args()
    tok_file = pathlib.Path(args.tok)
    n_ngram = args.ngram
    n_hl = args.n_hidden
    tok_data = read_file(tok_file)
    tok_data_split = []
    for sentence in tok_data:
        for token in sentence:
            tok_data_split.append(token.split("_"))
    tok_data = tok_data_split
    n_gram = create_ngrams(tok_data, n_ngram)
    vocab = create_vocab(tok_data)
    multi_hot_dict = multi_hot_encoding(vocab)
    X, Y = generate_x_y(multi_hot_dict, n_gram)
    model = create_model(n_hl)
    trained_model = train_model(model, X, Y)
    weights = get_weights(trained_model)
    embedding = get_embedding(vocab, weights)
    write_emb(embedding, output=args.output)



if __name__ == "__main__":
    main()