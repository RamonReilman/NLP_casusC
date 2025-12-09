from collections import Counter, defaultdict
import pickle
from typing import List, Tuple, Dict
import random

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
        :param path: Path to the text file.
        :return: List of lists of tokens.
        """
        vocab = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower() # so there is no conflict
                if not line:
                    continue
                words = line.split()
                for word in words:
                    vocab.append(list(word))  # Correctly saves as a list of character tokens
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

    def get_corpus_tokens(self, path: str) -> List[str]:
        """
        reads file, breaks into word
        :param path:
        :return:
        """
        all_tokens = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip().lower()
                if not line:
                    continue
                words = line.split()

                for word in words:
                    char_tokens = list(word)

                    bpe_tokens = self.encode(char_tokens)

                    all_tokens.extend(bpe_tokens)
        return all_tokens


    def training_ngram_model(self, path: str, n: int = 3):
        corpus = self.get_corpus_tokens(path)

        self.n = n
        self.ngram_model = defaultdict(Counter)

        for i in range(len(corpus) - n + 1):
            # The final token is the final element so no list out of index error
            context = tuple(corpus[i:i + n - 1])

            next_token = corpus[i + n - 1]

            self.ngram_model[context][next_token] += 1

    def generate_text(self, start_text: str, max_length: int = 50) -> str:

        context_size = self.n -1

        initial_char_tokens = list("".join(start_text.split()))
        initial_bpe_tokens = self.encode(initial_char_tokens)

        current_context = tuple(initial_bpe_tokens[-context_size:])
        generated_tokens = list(current_context)

        for i in range(max_length):
            if current_context in self.ngram_model:
                next_token_count = self.ngram_model[current_context]

                tokens, weights = zip(*next_token_count.items())

                next_token = random.choice(tokens, weights=weights, k=1)[0]

                generated_tokens.append(next_token)

                current_context = current_context[1:] + (next_token, )

        final_text = self.decode(generated_tokens)
        return final_text