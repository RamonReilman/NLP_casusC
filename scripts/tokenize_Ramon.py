import argparse
import pathlib


def to_path(path):
    pathed = pathlib.Path(path)
    if pathed.exists():
        return pathed
    raise FileNotFoundError(path)


def setup_parser():
    parser = argparse.ArgumentParser(
        prog="Tokenize",
        description="For all your tokenizing needs",
        epilog="help"
    )
    subparser = parser.add_subparsers(dest="subcommand")
    enc_gen = subparser.add_parser("generate-enc", description="Will generate a .enc file based on txt file input")
    enc_gen.add_argument("--txt_file", type=to_path,
                         help="Path to input .txt file", required=True)
    enc_gen.add_argument("-o", "--output", default="./output.enc", help="Path to generate an encoding file")
    enc_gen.add_argument("-m", "--max_merges", default=10, help="Maximum amount of merges to be generated", type=int)

    tok_gen = subparser.add_parser("generate-toc", description="Will generate a tokens for a text file using encoding")
    tok_gen.add_argument("--txt_file", type=to_path,
                         help="Path to input .txt file", required=True)
    tok_gen.add_argument("--enc_file", type=to_path,
                         help="Path to input .enc file", required=True)
    tok_gen.add_argument("-o", "--output", default="./output.tok", help="Path to generate a tok file")

    txt_gen = subparser.add_parser("generate-txt",
                                   description="Will generate a text file based on tokens and .enc file")
    txt_gen.add_argument("--enc_file", type=to_path,
                         help="Path to input .enc file", required=True)
    txt_gen.add_argument("--tok_file", type=to_path,
                         help="Path to input .tok file", required=True)
    txt_gen.add_argument("-o", "--output", default="./output.txt", help="Path to generate a txt file")
    return parser


def read_file(path):
    file = [] if path.suffix in (".txt", ".tok") else {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if path.suffix == ".txt":
                line = line.replace(" ", "_")
                file.append([char for char in line])
            if path.suffix == ".enc":
                line_split = line.split(" ")
                file[(line_split[0], line_split[1])] = line_split[-1]
            if path.suffix == ".tok":
                line_split = line.split(" ")
                file.append(line_split)
    return file


def init_vocab(corpus):
    vocab = set(char for word in corpus for char in word)
    return vocab


def init_freq(corpus):
    vocab = {}
    for sentence in corpus:
        for letter in sentence:
            if letter not in vocab:
                vocab[letter] = 0
            vocab[letter] += 1
    return vocab


def find_highest_pair(corpus):
    pairs_count = {}

    for sentence in corpus:
        n_sentence = len(sentence)
        for i in range(0, n_sentence - 1):

            first = sentence[i]
            second = sentence[i + 1]
            combined = (first, second)
            if first == ' ' or second == ' ':
                continue
            if combined not in pairs_count:
                pairs_count[combined] = 0
            pairs_count[combined] += 1

    if len(pairs_count) == 0:
        return None
    highest_pair_val = max(pairs_count.values())
    highest_pair = max(pairs_count.items(), key=lambda x: x[1])[0]
    return highest_pair, highest_pair_val


def update_freq(freq, highest_pair):
    pair = highest_pair[0]
    freq[f"{pair[0]}{pair[1]}"] = highest_pair[1]
    for i in range(0, highest_pair[1]):
        for j in range(0, 2):
            try:
                freq[pair[j]] -= 1
                if freq[pair[j]] == 0:
                    freq.pop(pair[j])
            except Exception as _:
                continue

    return freq


def update_corpus(corpus, highest_pair):
    highest_pair = highest_pair[0]
    new_corpus = []
    n = len(corpus)
    i = 0
    for sentence in corpus:
        n = len(sentence)
        i = 0
        temp_sentence = []
        while i < n:
            if i < n - 1 and ((sentence[i], sentence[i + 1]) == highest_pair):

                temp_sentence.append(f"{sentence[i]}{sentence[i + 1]}")
                i += 2
            else:
                temp_sentence.append(sentence[i])
                i += 1
        new_corpus.append(temp_sentence)
    return new_corpus


def update_vocab(new_token, vocab):
    vocab.add(f"{new_token[0][0]}{new_token[0][1]}")
    return vocab


def write_enc(merges, output):
    output = pathlib.Path(output)
    if output.is_dir():
        output = output / "output.enc"
    with open(output, "w", encoding="utf-8") as f:
        for merge1, merge2 in merges:
            f.write(f"{merge1} {merge2} {merge1}{merge2}\n")


def write_tok(tokens, output):
    output = pathlib.Path(output)
    if output.is_dir():
        output = output / "output.tok"

    with open(output, "w", encoding="utf-8") as f:
        for sentence in tokens:
            f.write(" ".join(sentence) + "\n")


def generate_enc(args):
    corpus = read_file(args.txt_file)
    vocab = init_vocab(corpus)
    freq = init_freq(corpus)
    tries = 0
    merges = []
    while len(merges) < args.max_merges and tries < 100000:
        highest_pair = find_highest_pair(corpus, )
        if highest_pair is None:
            break
        corpus = update_corpus(corpus, highest_pair)
        freq = update_freq(freq, highest_pair)
        vocab = update_vocab(highest_pair, vocab)
        merges.append(highest_pair[0])
        tries += 1
    write_enc(merges, args.output)


def update_tokens_with_mergerules(corpus, rules):
    new_corpus = []
    for sentence in corpus:
        changed = True
        while changed:
            changed = False
            n = len(sentence)
            i = 0
            new_sentence = []
            while i < n:
                if i < n - 1:
                    merge = get_merge(rules, sentence[i], sentence[i+1])
                    if merge:
                        new_sentence.append(rules.get((sentence[i], sentence[i + 1])))
                        i += 2
                        changed = True
                        continue

                new_sentence.append(sentence[i])
                i += 1
            sentence = new_sentence
        new_corpus.append(new_sentence)
    return new_corpus

def get_merge(rules, s1, s2):
    pair = (s1,s2)
    if s1 in rules.keys():
        return pair
    elif pair in rules.keys():
        return rules.get(pair)
    return None



def generate_toc(args):
    corpus = read_file(args.txt_file)
    enc = read_file(args.enc_file)
    tokens = update_tokens_with_mergerules(corpus, enc)
    write_tok(tokens, args.output)

def write_txt(text, output):
    output = pathlib.Path(output)
    if output.is_dir():
        output = output / "output.txt"
    with open(output, "w", encoding="utf-8") as f:
        for line in text:
            f.write("".join(line).replace("_", " ") + "\n")


def generate_txt(args):
    enc = read_file(args.enc_file)
    enc = {v: k for k, v in enc.items()}
    tokens = read_file(args.tok_file)
    n = []
    for sentence in tokens:
        changed = True
        while changed:
            changed = False
            new_sentence = []

            for tok in sentence:
                if tok in enc:
                    left, right = enc[tok]
                    new_sentence.extend([left, right])
                    changed = True
                else:
                    new_sentence.append(tok)

            sentence = new_sentence
        n.append(sentence)
    write_txt(n, args.output)



def main():
    args = setup_parser().parse_args()
    if args.subcommand == "generate-enc":
        generate_enc(args)
    elif args.subcommand == "generate-toc":
        generate_toc(args)
    elif args.subcommand == "generate-txt":
        generate_txt(args)


if __name__ == "__main__":
    main()
