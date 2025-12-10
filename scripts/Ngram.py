# ngram.py
import argparse
import sys
import os
from nlp_jarno import NgramModel


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using N-gram stats model",
    )

    parser.add_argument("-n", "--order", type=int, required=True, help="the order of the N-gram model (example. 3)")
    parser.add_argument("-i", "--input", nargs="+", required=True, help="one or more .tok files as input")
    parser.add_argument("-o", "--output", required=True, help="filename for the output (.tok)")
    parser.add_argument("-l", "--length", type=int, required=True, help="ammount of tokens to generate")

    args = parser.parse_args()

    all_tokens = []
    print(f"reading file")
    for file_path in args.input:
        if not os.path.exists(file_path):
            print(f"Warning file does not exist: {file_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
            all_tokens.extend(tokens)

    if not all_tokens:
        sys.exit("Error not all tokens found")


    model = NgramModel(args.order)
    model.train(all_tokens)

    generated_tokens = model.generate(args.length)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("\n".join(generated_tokens))
    print(f"Done! Resultaat saved in '{args.output}'")



if __name__ == "__main__":
    main()