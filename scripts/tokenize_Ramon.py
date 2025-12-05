import argparse
import pathlib

def to_path(path):
    pathed = pathlib.Path(path)
    if pathed.exists():
        return pathed
    raise FileNotFoundError(path)

def setup_parser():
    parser = argparse.ArgumentParser(
        prog = "Tokenize",
        description="For all your tokenizing needs",
        epilog="help"
    )
    subparser = parser.add_subparsers(dest = "subcommand")
    enc_gen = subparser.add_parser("generate-enc", description="Will generate a .enc file based on txt file input")
    enc_gen.add_argument("--txt_file", type=to_path,
                        help = "Path to input .txt file", required=True)
    enc_gen.add_argument("-o", "--output", default="./output.enc", help = "Path to generate an encoding file")
    enc_gen.add_argument("-m", "--max_merges", default=10, help="Maximum amount of merges to be generated", type=int)

    tok_gen = subparser.add_parser("generate-toc", description="Will generate a tokens for a text file using encoding")
    tok_gen.add_argument("--txt_file", type=to_path,
                        help = "Path to input .txt file", required=True)
    tok_gen.add_argument("--enc_file", type=to_path,
                        help = "Path to input .enc file", required=True)
    tok_gen.add_argument("-o", "--output", default="./output.tok", help = "Path to generate a tok file")
    return parser

def read_file(path):
    print(path)
    file = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if path.suffix == ".txt":
                letters = [char for char in line.strip()]
                file.extend(letters)
            if path.suffix == ".enc":

                line_split = line.strip().split(" ")
                file.append((line_split[0], line_split[1]))
    return file

def init_vocab(string):
    return set(string)

def init_freq(string):
    vocab = {}
    for letter in string:
        if letter not in vocab:
            vocab[letter] = 0
        vocab[letter] += 1
    return vocab

def find_highest_pair(tokens):
    pairs_count = {}
    n = len(tokens)

    for i in range(0, n-1):
        first = tokens[i]
        second = tokens[i+1]
        combined = (first, second)
        if first == ' ' or second == ' ':
            continue
        if combined not in pairs_count:
            pairs_count[combined] = 0
        pairs_count[combined] += 1
    if len(pairs_count) == 0:
        return None
    highest_pair_val = max(pairs_count.values())
    highest_pair = max(pairs_count, key=pairs_count.get)
    return highest_pair, highest_pair_val

def update_freq(freq, highest_pair):
    pair = highest_pair[0]
    freq[f"{pair[0]}{pair[1]}"] = highest_pair[1]
    for i in range(0, highest_pair[1]):
        for j in range(0, 2):
            try:
                freq[pair[j]]-=1
                if freq[pair[j]] == 0:
                    freq.pop(pair[j])
            except Exception as _:
                continue


    return freq

def update_tokens(tokens, highest_pair):
    highest_pair = f"{highest_pair[0][0]}{highest_pair[0][1]}"
    new_tokens = []
    n = len(tokens)
    j = 0
    for i in range(0, n-1):
        if j >= n-1:
            break
        combines = f"{tokens[j]}{tokens[j+1]}"
        if combines == highest_pair:
            new_tokens.append(highest_pair)
            j+=2
        else:
            new_tokens.append(tokens[j])
            j+=1
    if j < n:
        new_tokens.append(tokens[-1])

    return new_tokens
def update_vocab(new_token, vocab):
    vocab.add(f"{new_token[0][0]}{new_token[0][1]}")
    return vocab

def write_enc(merges, output):
    output = pathlib.Path(output)
    if output.is_dir():
        output = output / "output.enc"
    with open(output, "w", encoding="utf-8") as f:
        for merge1, merge2 in merges:
            f.write(f"{merge1} {merge2}\n")


def generate_enc(args):
    tokens = read_file(args.txt_file)
    vocab = init_vocab(tokens)
    freq = init_freq(tokens)
    tries = 0
    merges = []
    while len(merges) < args.max_merges and tries < 1000.000:
        highest_pair = find_highest_pair(tokens, )
        if highest_pair is None:
            break
        tokens = update_tokens(tokens, highest_pair)
        freq = update_freq(freq, highest_pair)
        vocab = update_vocab(highest_pair, vocab)
        merges.append(highest_pair[0])
        tries+=1
    write_enc(merges, args.output)

def generate_toc(args):
    tokens = read_file(args.txt_file)
    enc = read_file(args.enc_file)


    print(tokens)
def main():
    args = setup_parser().parse_args()
    print(args)
    if args.subcommand == "generate-enc":
        generate_enc(args)
    elif args.subcommand == "generate-toc":
        generate_toc(args)

if __name__ == "__main__":
    main()