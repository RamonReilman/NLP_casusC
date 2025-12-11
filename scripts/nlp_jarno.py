import math
import pathlib
from collections import Counter, defaultdict
import pickle
from typing import List, Tuple, Dict
import random

import numpy as np
from sklearn.neural_network import MLPClassifier


import pathlib

class BPETokenizer:
    def __init__(self):
        pass


    @staticmethod
    def read_file(path):
        file = []
        path = pathlib.Path(path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if path.suffix == ".txt":
                    line = line.replace(" ", "_")
                    file.append([char for char in line])
                elif path.suffix == ".enc":
                    split = line.split(" ")
                    file.append((split[0], split[1], split[2]))
                elif path.suffix == ".tok":
                    file.append(line.split(" "))
        return file

    @staticmethod
    def write_enc(merges, output):
        output = pathlib.Path(output)
        if output.is_dir():
            output = output / "output.enc"
        with open(output, "w", encoding="utf-8") as f:
            for merge1, merge2 in merges:
                f.write(f"{merge1} {merge2} {merge1}{merge2}\n")

    @staticmethod
    def write_tok(tokens, output):
        output = pathlib.Path(output)
        if output.is_dir():
            output = output / "output.tok"
        with open(output, "w", encoding="utf-8") as f:
            for sentence in tokens:
                f.write(" ".join(sentence) + "\n")

    @staticmethod
    def write_txt(text, output):
        output = pathlib.Path(output)
        if output.is_dir():
            output = output / "output.txt"
        with open(output, "w", encoding="utf-8") as f:
            for line in text:
                f.write("".join(line).replace("_", " ") + "\n")


    @staticmethod
    def init_vocab(corpus):
        return set(char for word in corpus for char in word)

    @staticmethod
    def init_freq(corpus):
        vocab = {}
        for sentence in corpus:
            for letter in sentence:
                vocab[letter] = vocab.get(letter, 0) + 1
        return vocab

    @staticmethod
    def find_highest_pair(corpus):
        pairs_count = {}
        for sentence in corpus:
            for i in range(len(sentence) - 1):
                a, b = sentence[i], sentence[i + 1]
                if a == ' ' or b == ' ':
                    continue
                pair = (a, b)
                pairs_count[pair] = pairs_count.get(pair, 0) + 1

        if not pairs_count:
            return None

        highest_pair = max(pairs_count.items(), key=lambda x: x[1])
        return highest_pair[0], highest_pair[1]

    @staticmethod
    def update_freq(freq, highest_pair):
        pair, count = highest_pair
        merged = f"{pair[0]}{pair[1]}"
        freq[merged] = count
        for _ in range(count):
            for j in range(2):
                if pair[j] in freq:
                    freq[pair[j]] -= 1
                    if freq[pair[j]] == 0:
                        freq.pop(pair[j])
        return freq

    @staticmethod
    def update_corpus(corpus, highest_pair):
        pair = highest_pair[0]
        new_corpus = []
        for sentence in corpus:
            temp = []
            i = 0
            while i < len(sentence):
                if i < len(sentence) - 1 and (sentence[i], sentence[i + 1]) == pair:
                    temp.append(sentence[i] + sentence[i + 1])
                    i += 2
                else:
                    temp.append(sentence[i])
                    i += 1
            new_corpus.append(temp)
        return new_corpus

    @staticmethod
    def update_vocab(merge, vocab):
        a, b = merge[0]
        vocab.add(a + b)
        return vocab

    @staticmethod
    def update_tokens_with_mergerules(corpus, rules):
        for first, second, merged in rules:
            for i, sentence in enumerate(corpus):
                new_sentence = []
                skip = 0
                for j in range(len(sentence)):
                    if skip:
                        skip -= 1
                        continue
                    if j < len(sentence) - 1 and sentence[j] == first and sentence[j+1] == second:
                        new_sentence.append(merged)
                        skip = 1
                    else:
                        new_sentence.append(sentence[j])
                corpus[i] = new_sentence
        return corpus


    def generate_enc(self, txt_file, max_merges=10, output="./output.enc"):
        corpus = self.read_file(txt_file)
        vocab = self.init_vocab(corpus)
        freq = self.init_freq(corpus)

        merges = []
        tries = 0

        while len(merges) < max_merges and tries < 100000:
            highest = self.find_highest_pair(corpus)
            if highest is None:
                break

            corpus = self.update_corpus(corpus, highest)
            freq = self.update_freq(freq, highest)
            vocab = self.update_vocab(highest, vocab)
            merges.append(highest[0])

            tries += 1

        self.write_enc(merges, output)

    def generate_toc(self, txt_file, enc_file, output="./output.tok"):
        corpus = self.read_file(txt_file)
        enc = self.read_file(enc_file)
        tokens = self.update_tokens_with_mergerules(corpus, enc)
        self.write_tok(tokens, output)

    def generate_txt(self, enc_file, tok_file, output="./output.txt"):
        enc = self.read_file(enc_file)
        enc_map = {merged: (a, b) for (a, b, merged) in enc}
        tokens = self.read_file(tok_file)

        final = []

        for sentence in tokens:
            changed = True
            while changed:
                changed = False
                new_sentence = []

                for tok in sentence:
                    if tok in enc_map:
                        left, right = enc_map[tok]
                        new_sentence.extend([left, right])
                        changed = True
                    else:
                        new_sentence.append(tok)

                sentence = new_sentence

            final.append(sentence)

        self.write_txt(final, output)



class NgramModel:
    def __init__(self, n: int):
        self.n = n
        self.model: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)

    def train(self, tokens: List[str]):


        if len(tokens) < self.n:
            return

        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i : i + self.n - 1])
            next_token = tokens[i + self.n - 1]
            self.model[context][next_token] += 1

    def generate(self, length: int) -> List[str]:

        current_context = random.choice(list(self.model.keys()))
        output = list(current_context)

        for i in range(length):

            if current_context in self.model:
                possible_next = self.model[current_context]
                tokens, counts = zip(*possible_next.items())

                next_token = random.choices(tokens, weights=counts, k=1)[0]

            else:
                next_token = random.choice(list(self.model.keys()))[0]

            output.append(next_token)
            current_context = tuple(output[-(self.n - 1):])
        return output
class Embedder:
    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
    def generate_token_dict(vocab):
        return {word: i for i, word in enumerate(vocab)}

    @staticmethod
    def multi_hot_encoding(vocab):
        token_dict = Embedder.generate_token_dict(vocab)
        multi_hot = np.zeros((len(vocab), len(vocab)))
        for i in range(len(vocab)):
            multi_hot[i][i] = 1

        return {vocab[i]: multi_hot[i] for i in range(len(vocab))}

    @staticmethod
    def create_vocab(tokens):
        vocab = sorted(list(set(token for sentence in tokens for token in sentence)))
        return vocab

    @staticmethod
    def generate_x_y(multi_hot_dict, n_grams):
        X = []
        Y = []
        for sentence in n_grams:
            for gram in sentence:
                temp_x = np.zeros(len(multi_hot_dict))
                n = len(gram)
                mid = math.floor(n/2)
                Y.append(multi_hot_dict.get(gram[mid]))
                for idx in range(n):
                    if idx != mid:
                        temp_x += multi_hot_dict.get(gram[idx])
                X.append(np.clip(temp_x, 0, 1))
        return X, Y

    @staticmethod
    def create_model(n_neurons):
        return MLPClassifier(hidden_layer_sizes=(n_neurons,))

    @staticmethod
    def train(model, X, Y):
        model.fit(X, Y)
        return model

    @staticmethod
    def get_weights(model):
        return model.coefs_[1].T

    @staticmethod
    def get_embedding(vocab, weights):
        token_dict = Embedder.generate_token_dict(vocab)
        return {token: weights[token_dict[token]] for token in vocab}

    @staticmethod
    def write_emb(embedding, output_path):
        with open(output_path, "w") as f:
            for token, vector in embedding.items():
                line = token + "".join(f" {v}" for v in vector) + "\n"
                f.write(line)

    def embed(self, tok_file, n_ngram=2, n_hidden=5, output="output.emb"):
        tok_data = BPETokenizer.read_file(pathlib.Path(tok_file))

        tok_data_split = []
        for sentence in tok_data:
            for token in sentence:
                tok_data_split.append(token.split("_"))

        tok_data = tok_data_split

        n_gram = self.create_ngrams(tok_data, n_ngram)
        vocab = self.create_vocab(tok_data)
        multi_hot = self.multi_hot_encoding(vocab)

        X, Y = self.generate_x_y(multi_hot, n_gram)

        try:
            model = self.create_model(n_hidden)
            trained_model = self.train(model, X, Y)
        except ValueError:
            print("No n-gram generated. Increase merges in the .enc file and regenerate tokens.")
            return

        weights = self.get_weights(trained_model)
        embedding = self.get_embedding(vocab, weights)
        self.write_emb(embedding, output)
