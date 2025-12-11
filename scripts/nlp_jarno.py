import math
import pathlib
from collections import Counter, defaultdict
import pickle
from typing import List, Tuple, Dict
import random

import numpy as np
from sklearn.neural_network import MLPClassifier

from Tokenize import read_file


class BPETokenizer:
    def __init__(self):
        # Data structures for the BPE encoding
        self.merges_order: List[Tuple] = []  # List of pairs in the order they were learned
        self.merges_map: Dict[Tuple[str, str], str] = {}  # Map: pair -> new_token

        # Placeholder for final token-to-ID mapping (if needed later)
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}

        self.n = 3
        self.ngram_model: Dict[Tuple, Counter] = {}

    @staticmethod
    def read_file(path: str) -> List[List[str]]:
        """
        Reads a text file and returns a list of lists of tokens.
        """
        vocab = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                if not line:
                    continue
                # NIEUW/FIX: Hier zou je idealiter een </w> token toevoegen of spaties
                # anders behandelen om de zinstructuur te behouden.
                # Voor nu laten we het zoals het was in jouw originele code.
                words = line.split()
                for word in words:
                    vocab.append(list(word))
        return vocab

    def get_stats(self, vocab: List[List[str]]) -> Counter:
        """
        Gets statistics about the vocabulary. How many pairs are there giving a counter dict back
        :param vocab: Vocabulary.
        :return: Counter with pairs of tokens.
        """
        # a subclass of dict that's specially designed for counting hashable objects in Python
        counts = Counter()
        for tokens in vocab:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                counts[pair] += 1
        return counts

    def merge(self, tokens: List[str], best_pair: Tuple[str, str], new_token: str) -> List[str]:
        """
        merge the tokens if they are equal to best pair. and return the results in the form of a list
        that contains the tokens that are merged.
        :param tokens: List of tokens.
        :param best_pair: Tuple of two tokens.
        :param new_token: New token.
        :return: List of merged tokens.
        """
        i = 0
        result = []
        token_len = len(tokens)

        while i < token_len:
            if i + 1 < token_len and (tokens[i], tokens[i + 1]) == best_pair:
                result.append(new_token)
                i += 2  # Skip the next token since it was just consumed
            else:
                result.append(tokens[i])
                i += 1
        return result

    def train(self, path: str, num_merges: int) -> Tuple[List, Dict]:
        """
        reads the text input, clear the merges_order and merges_map. For the amount of merges
        get the starts for the text. look in stats for the best pair using most_comon.
        make the new_token the best pair and update the voacb
        :param path: Path to the text file.
        :param num_merges: Number of merges.
        :return: List of merged tokens and merges_map.
        """
        initial_vocab = self.read_file(path)
        current_vocab = initial_vocab

        self.merges_order = []
        self.merges_map = {}

        for i in range(num_merges):
            stats = self.get_stats(current_vocab)
            if not stats:
                print(f"No more pairs to merge after {i} steps.")
                break

            best_pair, count = stats.most_common(1)[0]
            new_token = "".join(best_pair)

            self.merges_order.append(best_pair)
            self.merges_map[best_pair] = new_token

            # Update the vocabulary for the next iteration
            new_vocab = [self.merge(word_tokens, best_pair, new_token) for word_tokens in current_vocab]
            current_vocab = new_vocab

        return self.merges_order, self.merges_map

    def encode(self, tokens: List[str]) -> List[str]:
        """
        Encodes a list of tokens. by looping over the pairs in merges_order.
        then merging them repeatedly until no pairs are left
        then return a list of merged tokens.

        :param tokens: List of tokens.
        :return: List of encoded tokens.
        """
        current_tokens = tokens

        for pair in self.merges_order:
            new_token = self.merges_map[pair]

            # Use the merge function repeatedly until no more pairs are found
            current_tokens = self.merge(current_tokens, pair, new_token)

        return current_tokens

    def decode(self, tokens: List[str]) -> str:
        """
        Decodes a list of tokens. by looping over the pairs in merges_order.

        :param tokens:
        :return:
        """
        text = " ".join(tokens)  # Join tokens with a space

        # Iterate through the merges in REVERSE order
        for pair in reversed(self.merges_order):
            original_chars = pair[0] + " " + pair[1]  # e.g., "h e"
            new_token = self.merges_map[pair]  # e.g., "he"

            text = text.replace(new_token, original_chars)

        return text.replace(" ", "")  # Remove spaces to get back the original clean text

    def save_encoding(self, path: str):
        """

        :param path:
        :return:
        """
        encoding_data = {
            'merges_order': self.merges_order,
            'merges_map': self.merges_map,
        }
        with open(path, "wb") as f:
            pickle.dump(encoding_data, f)

    def load_encoding(self, path: str):
        """

        :param path:
        :return:
        """
        with open(path, "rb") as f:
            encoding = pickle.load(f)

        self.merges_order = encoding['merges_order']
        self.merges_map = encoding['merges_map']
        print(f"Loaded {len(self.merges_order)} merges from {path}")


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
        tok_data = read_file(pathlib.Path(tok_file))

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
