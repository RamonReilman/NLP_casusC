from collections import Counter
import pickle
from typing import List, Tuple, Dict

class BPETokenizer:
    def __init__(self):
        # Data structures for the BPE encoding
        self.merges_order: List[Tuple] = []  # List of pairs in the order they were learned
        self.merges_map: Dict[Tuple[str, str], str] = {}  # Map: pair -> new_token

        # Placeholder for final token-to-ID mapping (if needed later)
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}

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