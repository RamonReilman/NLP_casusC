from collections import Counter, defaultdict
import pickle
from typing import List, Tuple, Dict
import random
import re
from math import log

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
    def generate_pair_count(corpus):
        pair_counts = defaultdict(int)
        for sentence in corpus:
            for i in range(len(sentence) - 1):
                pair_counts[(sentence[i], sentence[i+1])] += 1
        return pair_counts

    @staticmethod
    def find_highest_pair(pair_count):
        if not pair_count:
            return None
        return max(pair_count.items(), key=lambda x: x[1])


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
    def update_pair_count(x,y,a,b,pair_counts):
        ab = a+b
        pair_counts[(a, b)] -= 1
        if pair_counts[(a, b)] <= 0:
            pair_counts.pop((a, b), None)
        if x is not None:
            pair_counts[(x, a)] = max(0, pair_counts.get((x, a), 0) - 1)
        if y is not None:
            pair_counts[(b, y)] = max(0, pair_counts.get((b, y), 0) - 1)

        # increment new pairs
        if x is not None:
            pair_counts[(x, ab)] = pair_counts.get((x, ab), 0) + 1
        if y is not None:
            pair_counts[(ab, y)] = pair_counts.get((ab, y), 0) + 1
        return pair_counts


    def update_corpus(self, corpus, highest_pair, pair_count):
        pair = highest_pair[0]
        new_corpus = []
        for sentence in corpus:
            temp = []
            i = 0
            while i < len(sentence):
                if i < len(sentence) - 1 and (sentence[i], sentence[i + 1]) == pair:
                    x = temp[-1] if temp else None
                    y = sentence[i + 2] if i + 2 < len(sentence) else None
                    pair_count = self.update_pair_count(x,y,sentence[i],sentence[i+1], pair_count)
                    temp.append(sentence[i] + sentence[i + 1])
                    i += 2
                else:
                    temp.append(sentence[i])
                    i += 1
            new_corpus.append(temp)
        return new_corpus, pair_count

    def update_vocab(self, merge, vocab):
        a, b = merge[0]
        vocab.add(a + b)
        return vocab

    def update_tokens_with_mergerules(self, corpus, rules):
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


    def generate_enc(self, txt_file, max_merges, output="./output.enc"):
        print("Generating corpus and vocab")
        corpus = self.read_file(txt_file)
        vocab = self.init_vocab(corpus)
        pair_count = self.generate_pair_count(corpus)
        merges = []

        print("Starting iteration")
        for i in range(0, max_merges):
            highest = self.find_highest_pair(pair_count)
            if highest is None:
                break

            corpus, pair_count = self.update_corpus(corpus, highest, pair_count)
            vocab = self.update_vocab(highest, vocab)
            merges.append(highest[0])

        print(f"Writing enc file to: {output}")
        self.write_enc(merges, output)

    def generate_toc(self, txt_file, enc_file, output="./output.tok"):
        print("Loading corpus and enc file")
        corpus = self.read_file(txt_file)
        enc = self.read_file(enc_file)
        print("Generating tokens")
        tokens = self.update_tokens_with_mergerules(corpus, enc)
        print(f"Writing output to: {output}")
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


# ------------------------BagOfWords-------------------------------

class Bagofwords:
    def __init__(self, tokens, encoding_type="multi_hot"):
        if encoding_type not in ["multi_hot", "frequency", "tf_idf"]:
            raise ValueError("encoding_type unknown: " + encoding_type)
        self.tokens = tokens
        self.tot_counts = {i: 0 for i in tokens.keys()}
        self.encoding_type = encoding_type

    def create_bag(self, X):
        X_tokens = []
        X_encoded = []

        # convert strings to tokens
        for x_i in X:
            X_tokens.append(self.string_to_tokens(x_i))

        # use encoding based on given encoding type
        local_counts = {i: 0 for i in self.tot_counts.keys()}
        for x_i_token in X_tokens:
            for token in x_i_token:
                local_counts[token] = local_counts[token] + 1
        if self.encoding_type == "multi_hot":
            X_encoded = self.multi_hot(local_counts)
        elif self.encoding_type == "frequency":
            X_encoded = self.frequency(local_counts)
        elif self.encoding_type == "tf_idf":
            X_encoded = self.tf_idf(local_counts)
        return X_encoded

    @staticmethod
    def multi_hot(token_counts):
        return [int(bool(count)) for count in token_counts.values()]

    @staticmethod
    def frequency(token_counts):
        tot_count = sum(token_counts.values())
        return [0 if count == 0 else count / tot_count for count in token_counts.values()]

    def tf_idf(self, local_token_counts):
        full_tot_count = sum(self.tot_counts.values())
        local_tot_count = sum(local_token_counts.values())
        return [0 if local_count == 0 or full_count == 0 else
                (local_count / local_tot_count) * log(full_count / full_tot_count)
                for full_count, local_count in zip(self.tot_counts.values(), local_token_counts.values())]

    def fit(self, X):
        """
        fits data on given string of text (only necessary when using tf_idf)
        :param X: long string to fit
        """
        X_tokens = []

        for x_i in X:
            X_tokens.append(self.string_to_tokens(x_i))

        #
        for x_i_token in X_tokens:
            for token in x_i_token:
                self.tot_counts[token] = self.tot_counts[token] + 1

    def string_to_tokens(self, input_string):
        """
        generates token list based on this class' tokens from given string
        :param input_string: string to be converted to tokens
        :return: list of integers holding the tokens
        """
        # uses string-to-token converter of tokenize_Stijn, replace later with usage of nlp
        remaining_characters = input_string
        current_token = len(self.tokens.values()) - 1
        used_locations = []
        token_output = []
        while remaining_characters:
            # regex to check if only in-between characters remain in the string
            if not bool(re.compile(r'[^\n\t .,!?]').search(remaining_characters)) or current_token == -1:
                break
            found = remaining_characters.find(self.tokens[list(self.tokens.keys())[current_token]])
            if found == -1:
                current_token -= 1
            else:
                # append the found token to the right location and replace the found token in the string with blank space
                location = 0
                while True:
                    if location == len(used_locations) or found < used_locations[location]:
                        used_locations.insert(location, found)
                        token_output.insert(location, current_token)
                        remaining_characters = (remaining_characters[:found] + " " * len(self.tokens[current_token]) +
                                                remaining_characters[found + len(self.tokens[current_token]):])
                        break
                    location += 1
        return token_output

    def __str__(self):
        return f"provided_tokens: {self.tokens}, counted tokens in trainset: {self.tot_counts}"


