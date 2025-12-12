import argparse
import pathlib
from nlp_main_file import BPETokenizer


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


def main(args=None):
    args = setup_parser().parse_args(args)
    tokenizer = BPETokenizer()
    if args.subcommand == "generate-enc":
        txt_file = args.txt_file
        max_merges = args.max_merges
        output = args.output
        tokenizer.generate_enc(txt_file, max_merges, output)
    elif args.subcommand == "generate-toc":
        txt_file = args.txt_file
        enc_file = args.enc_file
        output = args.output
        tokenizer.generate_toc(txt_file, enc_file, output)
    elif args.subcommand == "generate-txt":
        enc_file = args.enc_file
        tok_file = args.tok_file
        output = args.output
        tokenizer.generate_txt(enc_file, tok_file, output)


if __name__ == "__main__":
    main()
