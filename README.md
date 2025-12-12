# Natural Language Processing casus

## Authors:

- Ramon - RamonReilman
- Stijn - DeStuiterBal
- Jarno - azzipxonraj

---

## Relevance

This package was used to train students on Natural Language Processing. It's main
goal was to develop different scripts based on assignments given by their teacher 
that resulted in this product.

## Description

This package is a try to manually replicate a Natural Language Processing Model,
There are sevaral scripts in the package that do different things like a tokenizer,
sentence generator and different training methods. The goal of this package was for a simple
and easy to understand NLP model to be created.

### Key functionalities

- Tokenization
- Bag-Of-Words classification
- N-gram text generation
- Embeddings

## System requirements and installation:

## Usage:

## Background info 

### Tokenization

Tokenization plays a fundamental part in Natural Language Processing. 
It divides a text input into smaller subunits called tokens. 
These tokens are the basic units (like words, subwords, or characters) that machine learning models consume.
It helps in improving the interpretability of text by different models (GeeksforGeeks, 2025a).

![image_example_tokenizer](/imgs/image-removebg-preview%20(5).png)

_Figure 1: Simpel Tokenizer example_

In figure 1, a simple explanation is shown on what a tokenizer does. In this package, 
we use Byte-Pair encoding.

This is a tokenization technique that breaks words down into smaller 
subwords that then get put in a loop where the most common pairs of characters are found and combined until the vocabulary reached 
the desired size .

**So how does Byte-Pair encoding work?**

Suppose we have the following string of four words: "ab", "bc", "bcd" and "cde". We can start by calculating the frequency of each byte (GeeksforGeeks, 2025b), 
also known as each character. This will result in the following vocabulary {"a", "b", "c", "d", "e"}

Where:

| Word | Count (Frequency) |
|------|-------------------|
| b    | 3                 |
| c    | 3                 |
| d    | 2                 |
| a    | 1                 |
| e    | 1                 |

_Table 1: byte pair frequency_

The most common pair also needs to be found, this appears to be bc. How did we find this? well, 
if we look at our previous string: "ab", "bc", "bcd" and "cde". 
It can be seen that bc appears next to each other twice. So because of this, 
we merge the pair to create a new subword. and we update the frequency.


| Word | Count (Frequency) |
|------|-------------------|
| b    | 3                 |
| c    | 3                 |
| d    | 2                 |
| a    | 1                 |
| e    | 1                 |
| bc   | 2                 |


_Table 2: byte pair frequency merged one time_

Besides the updated frequency we also update the vocabulary, because we have a new unique word.
This makes for the following vocabulary {"a", "b", "c", "d", "e", "bc"}.

Crucially, after this first merge, the corpus must be updated: Original Corpus: {"ab", "bc", "bcd", "cde"} 
Updated Corpus (replacing "bc" with the new token): {"a b", "bc", "bc d", "c d e"}

These steps of finding sub words based on pairs will continue for N (user input) times. 
This process continues iteratively until the desired vocabulary size is reached or no more pairs can be merged.

The final token vocabulary will be the set of all initial characters plus all the merged subword tokens created during the process.


### N-gram text generation

N-gram is a consecutive subsequence of N tokens, extracted from a text. Lets say we have this text: "Hello, This is an example"
this can be turned into 

- 1 gram (**Unigrams**) "Hello", "This", "is", "an", "example" 
- 2 gram (**Bigrams**) "Hello This", "This is", "is an", "an example" 
- 3 gram (**Trigrams**) "Hello This is", "This is an", "is an example" 
(More grams means more training text needed for better accuracy). 

An N-gram languague model is a statistical model that calculate the chacne of the next token.
The model decides what the next token (word) should be based on the previous N-tokens.
The larger the N value gets, the more context the model considers (This also means larger data input is needed).
The N-gram model needs to be trained, this can be done by giving it a Corpus 
(a large structured collection of authentic text or speech data, like books, articles
etc. GeeksforGeeks, 2025a). In the case of our package we will pre-process this text to force everything to be lowercase
and removing the punctuation and spaces.

**The mathematical formulas**

The basis of N-gram relies on the Bayes Theorem

$$P(A|B) = \frac{P(A \cap B )}{P(B)}$$

Where:

- P(A|B) = Probability of A when B is given.
- $P(A \cap B )$ = Probability of both A and B occuring
- P(B) = Probability of B

So how does this apply to N-gram?

Well it does by Bigram (2-gram) & Trigram (3-gram) probability, for this case only the Trigram formula will be shown.

$$P(t_i|t_{i-2},t_{i-1}) = \frac{count(t_{i-2},t_{i-1},t_i)}{count(t_{i-2},t_{i-1})}$$

Where:

- t = token 
- $t_{i-n}$ = token at place n
- count = a function that remembers the probability from a variable compared to the whole corpus.

_Note that making this formula bi gram is done by taking away $t_{i-2}$_

### Bag of Words

Text data needs to be converted to numbers so that a machine can use it and algorithms can understand it.
One method to do so, is the Bag of Words model. It turns text like sentences, paragraphs or full documents of text into  a multidimensional numerical vector.
It counts how often each word appears but then ignores the order of the words. The model does not care for order or the grammer, it focuses solely on counting how many times a word appears in the input (text, sentence, document etc.) (Wikipedia contributors, 2025).

The bag of words model can be used for a different amount of things, 
like text classification, sentiment analysis and clustering (GeeksforGeeks, 2025).

For Bag Of Words to work there are some key components.

**Vocabulary** : Reading through a file and saving each unique word into a vocabulary.

**Document as a Vector** : Each input file wll be represented as a vector. Each element show the frequency of the words from the vocabulary in that
input file. The frequency of each word is used as a feature for the model. Shown in table 3 is an example of how this could look like.

Example input: "Hello World, I love pizza"

| Word       | Count (Frequency) |
|------------|-------------------|
| Hello      | 1                 |
| World      | 1                 |
| I          | 1                 |
| love       | 1                 |
| Hamburgers | 0                 |
| Pizza      | 1                 |

_Table 3: Input file vector example_

An important step is **Pre-processing**, this means to process the text for unused characters in this case. 
This means removing reading characters like questions mark and other non-word characters, Removing extra spaces and
Converting the text to lowercase. 

The next step is **Counting words**, here we make the table 1 example. This can be stored in a dataframe or a dictionary.
The basis of this function comes down to: If it does not exist add word to the dictionary. If it does exist add 1 to the count of the word.

**The different methods for counting words**

| Method                                                       | What Does It Measure? | How Does It Work?                                                                  | Resulting Values                                                     | Key Advantage                                                               |
|--------------------------------------------------------------|-----------------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------|-----------------------------------------------------------------------------|
| 1. Multi-hot Encoding (Sometimes incorrectly called One-hot) | Presence              | Checks whether a token is or is not present in the document.                       | Sequence of 0 (not present) or 1 (present).                          | Simple and fast; preserves the length of the vocabulary.                    |
| 2. Frequency Encoding (Term Frequency)                       | Frequency/Volume      | Counts how many times a token appears in the document.                             | Integers (raw counts) or Floats (fraction, percentage of the total). | Gives weight to the most important/frequent words within this document.     |
| 3. TF-IDF (Term-Frequency Inverse-Document-Frequency)        | Importance/Relevance  | Weighs the frequency within this document against its rarity across all documents. | Float (a weighted score), often between 0 and $\approx$ 15.          | Assigns a high score to words that are unique and specific to the document. |

_Table 4: different counting methods_
Based on the following literature (GeeksforGeeks, 2025b), (GeeksforGeeks, 2025d), (GeeksforGeeks, 2025a).


The Last step is to select the vocabulary size, to manage the memory and computational complexity we limit the vocabulary to the top N (user chosen) most frequent words.
These top N words are from across all input files (if more then one were given). This reduced vocabulary then forms the final feature set.

Now u have the Top N most important words, these can be used for various machine learning related trainings.

**Why use bag of words?**

The model is simple, easily implementable and can be used for various natural language processing tasks. The results
can be visualised making them easy to interpret as well.

However, it comes of the cost of loosing context, and working wit large data sets can give efficiency issues.


### Embedding
While a computer knows how to use and work with integers, floats and other numerical data it does not know how natural language works. Sentences and words are not defined by the existence of them, but in the context in which they are used.
A computer simply does not understand this "freedom".
We still want to use the computing power of computers to work with words or sentences (or tokens). This can be done via embedding.

"A word embedding is a representation of a word" (“Word Embedding,” 2025)

#### Why embedding?
Words cannot be understood by a computer, but we can represent words, their usage and context via a "place" in the natural language space.
This space, and the coordinates of the words, can be used to cluster certain words, contexts, or synonyms together. Giving the computer an "understanding" of these words.
These can then be used for many things:

- Biology based research into topics, and learning said topics
- Generating summaries based on many articles and their abstracts

and more!

#### Implementation of embedding
Our implementation of token-embedding relies on our tokenizer: [see here](#Tokenization). Generated token are read and stored into a list. Tokens that are made up of several words are split up to get a more "word-like" embedding. This is done to increase clustering, and also helps with filtering specific words.
These tokens are transferred to a N2 + 1 gram. Which means that will generate a list with a word and the other N words (or tokens) surrounding it.
This is done to place tokens within a context instead of alone. 

A vocabulary is created, after the generation of the N2 + 1 grams. This vocabulary is a hashset of all tokens. Which is then used to generate a multi-hot-encoding matrix.

Our implementation uses multi-hot-encoded data as an input to generate the embedding. Multi-hot-encoding is slighty different compared to one-hot-encoding. One-hot-encoding is a vector that contains 0, and a maximum of one "1". That being a vector that characterizes a word into integers.
Our multi-hot-encoding works via the adding together of one-hot-encoded vectors, and clipping these between 0 and 1. This will generate a vector that characterizes the words in any given input sentence or n-gram.

##### Training
This multi-hot-encoding (without the middle word) will be given as an input to a MLPclassifier. This classifier has an input layer, the size of our vocabulary. 1 hidden layer with a given amount of neurons. And an output layer that gives the probability of a word given the n-gram.
The classifier is then trained to generate fitting words based on the given n-gram. This is done by managing the weights and biases of the model.

#### After training
The weights of the hidden layer are extracted from the model. These weights are the embedding of our tokens. The values represent the tokens in language space, and can be used to cluster, plot, or reduce the dimension (via PCA).
This is writen to a .emb file, with tokens and weights seperated by spaces.

(“Word Embedding,” 2025; Word Embeddings in NLP, 23:05:21+00:00)
## How to use this package

**Commands**

Commands for the tokenize function.

| Subcommand   | Action | Argument | Flag / Syntax    | Type | Default      | Required? | Description                                              |
|--------------|--------|----------|------------------|------|--------------|-----------|----------------------------------------------------------|
| generate-enc | Train  | Input    | --txt_file       | Path | -            | Yes       | Path to the source .txt file used to learn the encoding. |
|              |        | Output   | -o, --output     | Path | ./output.enc | No        | Path to save the generated .enc file.                    |
|              |        | Config   | -m, --max_merges | Int  | 10           | No        | Maximum amount of merges (vocabulary size control).      |
| generate-toc | Encode | Input    | --txt_file       | Path | -            | Yes       | Path to the .txt file you want to tokenize.              |
|              |        | Input    | --enc_file       | Path | -            | Yes       | Path to the .enc file (rules) to use for tokenization.   |
|              |        | Output   | -o, --output     | Path | ./output.tok | No        | Path to save the resulting token file.                   |
| generate-txt | Decode | Input    | --tok_file       | Path | -            | Yes       | Path to the .tok file you want to decode back to text.   |
|              |        | Input    | --enc_file       | Path | -            | Yes       | Path to the .enc file used to decode the tokens.         |
|              |        | Output   | -o, --output     | Path | ./output.txt | No        | Path to save the reconstructed text file.                |

_Table 5: Commands for the tokenize function explained._


Commands for the N-gram function

| Argument | Flag / Syntax | Type    | Default | Required? | Description                                   |
|----------|---------------|---------|---------|-----------|-----------------------------------------------|
| Order    | -n, --order   | Int     | -       | Yes       | The order of the N-gram model (e.g., 3).      |
| Input    | -i, --input   | Path(s) | -       | Yes       | One or more .tok files to use as source data. |
| Output   | -o, --output  | Path    | -       | Yes       | Filename for the generated output .tok file.  |
| Length   | -l, --length  | Int     | -       | Yes       | Total amount of tokens to generate.           |

_Table 6: Commands for the N-gram function explained._


Commands for the embed function

| Argument | Flag / Syntax   | Type | Default    | Required? | Description                                                         |
|----------|-----------------|------|------------|-----------|---------------------------------------------------------------------|
| Input    | -t, --tok       | Path | -          | Yes       | Path to the input .tok file.                                        |
| Window   | -n, --ngram     | Int  | 2          | No        | Window size of the 2N+1 gram (context size).                        |
| Output   | -o, --output    | Path | output.emb | No        | Path to write the resulting .emb file.                              |
| Neurons  | -nn, --n_hidden | Int  | 5          | No        | Number of neurons in the hidden layer (dimension of the embedding). |

_Table 7: Commands for the embed function explained._


# References

- GeeksforGeeks. (2025a, juli 11). NLP | Wordlist Corpus. GeeksforGeeks. https://www.geeksforgeeks.org/nlp/nlp-wordlist-corpus/
- GeeksforGeeks. (2025, 17 juli). Bag of words (BoW) model in NLP. GeeksforGeeks. https://www.geeksforgeeks.org/nlp/bag-of-words-bow-model-in-nlp/\
- Wikipedia contributors. (2025, 10 december). Bag-of-words model. Wikipedia. https://en.wikipedia.org/wiki/Bag-of-words_model
- GeeksforGeeks. (2025b, augustus 27). BytePair Encoding (BPE) in NLP. GeeksforGeeks. https://www.geeksforgeeks.org/nlp/byte-pair-encoding-bpe-in-nlp/
- GeeksforGeeks. (2025a, juli 11). Tokenization in NLP. GeeksforGeeks. https://www.geeksforgeeks.org/nlp/nlp-how-tokenizing-text-sentence-words-works/
- GeeksforGeeks. (2025b, juli 12). Feature encoding techniques machine learning. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/feature-encoding-techniques-machine-learning/
- GeeksforGeeks. (2025d, augustus 13). Understanding TFIDF (Term FrequencyInverse Document Frequency). GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/
- GeeksforGeeks. (2025a, juli 11). One hot encoding in machine learning. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/
- Word embedding. (2025). In Wikipedia. https://en.wikipedia.org/w/index.php?title=Word_embedding&oldid=1323019878
- Word Embeddings in NLP. (23:05:21+00:00). GeeksforGeeks. https://www.geeksforgeeks.org/nlp/word-embeddings-in-nlp/
