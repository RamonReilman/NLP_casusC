
import argparse
import os
import sys
from nlp_jarno import BPETokenizer, NgramModel

def setup_parser():
    parser = argparse.ArgumentParser(
        description="Byte-Pair Encoding (BPE) Tokenizer.",
    )
    subparsers = parser.add_subparsers(
        dest="subcommand",
        required=True
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument('input_file', type=str)
    train_parser.add_argument('output_enc', type=str)
    train_parser.add_argument('--num-merges', type=int, default=100)

    encode_parser = subparsers.add_parser("encode")
    encode_parser.add_argument('input_file', type=str)
    encode_parser.add_argument('encoding_file', type=str)
    encode_parser.add_argument('output_tok', type=str)

    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument('input_tok', type=str)
    decode_parser.add_argument('encoding_file', type=str)
    decode_parser.add_argument('output_file', type=str)

    ngram_parser = subparsers.add_parser("ngram")
    ngram_parser.add_argument("-n", "--order", type=int, required=True, help="the order of the N-gram model (example. 3)")
    ngram_parser.add_argument("-i", "--input", nargs="+", required=True, help="one or more .tok files as input")
    ngram_parser.add_argument("-o", "--output", required=True, help="filename for the output (.tok)")
    ngram_parser.add_argument("-l", "--length", type=int, required=True, help="ammount of tokens to generate")

    return parser


def main():
    args = setup_parser().parse_args()

    if args.subcommand == "ngram":
        all_tokens = []
        for input_file in args.input:
            if not os.path.exists(input_file):
                sys.exit("Input file '{}' does not exist.".format(input_file))
            with open(input_file, "r", encoding='utf-8') as f:
                all_tokens.extend([line.strip() for line in f if line.strip()])

        if not all_tokens:
            sys.exit("No tokens found in '{}'".format(args.input))

        model = NgramModel(args.order)
        model.train(all_tokens)

        generated = model.generate(args.length)

        with open(args.output, "w", encoding='utf-8') as f:
            f.write("\n".join(generated))
        print("Wrote output to '{}'".format(args.output))
        return

    tokenizer = BPETokenizer()

    print(f"Executing command: {args.subcommand}")

    if args.subcommand == "train":
        tokenizer.train(args.input_file, args.num_merges)
        tokenizer.save_encoding(args.output_enc)
        print(f"Encoding saved to {args.output_enc}")

    elif args.subcommand == "encode":
        tokenizer.load_encoding(args.encoding_file)

        raw_vocab = tokenizer.read_file(args.input_file)

        encoded_tokens = []
        for word_tokens in raw_vocab:
            encoded_tokens.extend(tokenizer.encode(word_tokens))

        output_string = "\n".join(encoded_tokens)
        with open(args.output_tok, 'w', encoding='utf-8') as f:
            f.write(output_string)
        print(f"Tokens saved to {args.output_tok}")

    elif args.subcommand == "decode":
        tokenizer.load_encoding(args.encoding_file)

        with open(args.input_tok, 'r', encoding='utf-8') as f:
            tokenized_text = [line.strip() for line in f if line.strip()]

        decoded_text = tokenizer.decode(tokenized_text)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(decoded_text)
        print(f"Decoded text saved to {args.output_file}")


if __name__ == "__main__":
    main()