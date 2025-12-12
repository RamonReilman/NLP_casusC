
import pathlib
from nlp_main_file import Embedder
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

def main():
    args = setup_parser().parse_args()
    input = args.tok
    n_ngram = args.ngram
    output = args.output
    n_hidden_neurons = args.n_hidden
    embedder = Embedder()
    embedder.embed(input, n_ngram, n_hidden_neurons, output)



if __name__ == "__main__":
    main()