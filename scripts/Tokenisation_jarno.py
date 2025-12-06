# tokenize.py

import argparse
import pickle
import sys
# Import the class from the nlp.py file
from nlp_jarno import BPETokenizer


def setup_parser():
    """Sets up the argument parser with subcommands for BPE."""
    parser = argparse.ArgumentParser(
        description="Byte-Pair Encoding (BPE) Tokenizer.",
    )
    # Use subparsers to handle the different modes
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

    return parser


def main():
    """Main execution function that handles command parsing and dispatch."""
    args = setup_parser().parse_args()
    tokenizer = BPETokenizer()

    print(f"Executing command: {args.subcommand}")

    if args.subcommand == "train":
        tokenizer.train(args.input_file, args.num_merges)
        tokenizer.save_encoding(args.output_enc)
        print(f"Encoding saved to {args.output_enc}")

    elif args.subcommand == "encode":
        tokenizer.load_encoding(args.encoding_file)

        # Read and prepare the raw text for encoding
        raw_vocab = tokenizer.read_file(args.input_file)

        # Flatten the list of lists into a single list of tokens for the encoder
        encoded_tokens = []
        for word_tokens in raw_vocab:
            # Encode each word's tokens and extend the list
            encoded_tokens.extend(tokenizer.encode(word_tokens))

        # Save tokens: one token per line
        output_string = "\n".join(encoded_tokens)
        with open(args.output_tok, 'w', encoding='utf-8') as f:
            f.write(output_string)
        print(f"Tokens saved to {args.output_tok}")

    elif args.subcommand == "decode":
        tokenizer.load_encoding(args.encoding_file)

        # Read tokens from the .tok file
        with open(args.input_tok, 'r', encoding='utf-8') as f:
            tokenized_text = [line.strip() for line in f if line.strip()]

        # Decode and save the output
        decoded_text = tokenizer.decode(tokenized_text)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(decoded_text)
        print(f"Decoded text saved to {args.output_file}")


if __name__ == "__main__":
    main()