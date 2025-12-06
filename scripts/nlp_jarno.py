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
