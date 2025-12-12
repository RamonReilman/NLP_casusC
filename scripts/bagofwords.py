from nlp_jarno import Bagofwords, BPETokenizer
import sys


def extract_abstracts(filepath):
    """
    reads file of format [paper title]: paper abstract \n
    :param filepath: string path to file
    :return: dictionary where key: string title of paper, value: string abstract of paper
    """
    output_dict = {}
    with open(filepath, "r") as train:
        for line in train.read().split("\n"):
            header, abstract = line.split("]:")
            abstract = abstract.replace(" ", "_")
            output_dict[header[1:]] = [char for char in abstract]
    return output_dict


def write_bow(output_dict, output_file="./output.bow"):
    """
    write dictionary to .bow file
    :param output_dict: dictionary with: paper_title: encripted list
    :param output_file: filepath to write to
    :return: genereates .bow file in format [paper_title]: encripted_numbers
    """
    with open(output_file, "w") as output:
        for key in output_dict.keys():
            output.write("[" + key + "]: " + ", ".join(str(i) for i in list(output_dict[key])) + "\n")


def get_tokens(text, max_merges=10):
    """

    :param text: list of text
    :param max_merges: number of times merges take place
    :return: token keys
    """
    tokenizer = BPETokenizer()
    corpus = text
    vocab = tokenizer.init_vocab(corpus)
    pair_count = tokenizer.generate_pair_count(corpus)

    merges = []
    tries = 0

    for i in range(0, max_merges):
        highest = tokenizer.find_highest_pair(pair_count)
        if highest is None:
            break

        corpus, pair_count = tokenizer.update_corpus(corpus, highest, pair_count)
        vocab = tokenizer.update_vocab(highest, vocab)
        merges.append(highest[0])

        tries += 1

    return vocab


def main():
    if sys.argv[1] == "--help" or sys.argv[1] == "-h":
        print("""
        Usage: bagofwords.py encoding_type inputfile [inputfile] output.bow
        
        params:
            encoding_type: [multi_hot | frequency | tf_idf]
            inputfile = text file with each line having: [title]: text
            output = .bow file with each line having: [title]: encoded string
        """)
        sys.exit()

    # extract commandline arguments
    encoding_type = sys.argv[1]
    inputs = sys.argv[2:-1]
    output_file = sys.argv[-1]

    all_abstracts = []
    abstract_dicts_list = []

    # read all input files
    for inputfile in inputs:
        extracted = extract_abstracts(inputfile)
        all_abstracts = all_abstracts + list(extracted.values())
        abstract_dicts_list.append(extracted)

    freq = get_tokens(all_abstracts, 100)
    token_input = {i: tok for i, tok in enumerate(freq)}

    bag = Bagofwords(token_input, encoding_type)

    X = "".join("".join(all_chars) for all_chars in all_abstracts)

    bag.fit(X)
    # loop through all abstacts and save them in output dict
    output = {}
    for dict in abstract_dicts_list:
        for key in dict.keys():
            output[key] = bag.create_bag(dict[key])
    write_bow(output, output_file)


if __name__ == "__main__":
    main()

