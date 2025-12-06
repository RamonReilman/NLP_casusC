from collections import Counter
import pickle
from typing import List, Tuple, Dict
import os
import re


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
        """Reads a .txt file, splits text into words, and tokenizes each word into characters."""
        vocab = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                if not line:
                    continue
                words = line.split()
                for word in words:
                    vocab.append(list(word))  # Correctly saves as a list of character tokens
        return vocab

    def get_stats(self, vocab: List[List[str]]) -> Counter:
        """Counts the frequency of all adjacent token pairs."""
        counts = Counter()
        for tokens in vocab:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                counts[pair] += 1
        return counts

    def merge(self, tokens: List[str], best_pair: Tuple[str, str], new_token: str) -> List[str]:
        """Performs one merge operation on a list of tokens."""
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
        """Performs BPE training on the text file at 'path'."""
        initial_vocab = self.read_file(path)  # Use the correct reader
        current_vocab = initial_vocab

        self.merges_order = []
        self.merges_map = {}

        for i in range(num_merges):
            stats = self.get_stats(current_vocab)  # FIX: STATS CALCULATION ADDED

            if not stats:
                print(f"No more pairs to merge after {i} steps.")
                break

            best_pair, count = stats.most_common(1)[0]  # FIX: Corrected variable name from counts to count
            new_token = "".join(best_pair)

            self.merges_order.append(best_pair)
            self.merges_map[best_pair] = new_token

            # Update the vocabulary for the next iteration
            new_vocab = [self.merge(word_tokens, best_pair, new_token) for word_tokens in current_vocab]
            current_vocab = new_vocab

        return self.merges_order, self.merges_map

    def encode(self, tokens: List[str]) -> List[str]:
        """Applies the learned merges to a list of tokens."""
        current_tokens = tokens

        # Loop through the merges in the exact order they were learned (from self)
        for pair in self.merges_order:  # FIX: Removed redundant args and used self.merges_order
            new_token = self.merges_map[pair]

            # Use the merge function repeatedly until no more pairs are found
            current_tokens = self.merge(current_tokens, pair, new_token)

        return current_tokens

