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

Tokenisation is essential in Natural Language Processing.

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
It counts how often each word appears but then ignores the order of the words. The model does not care for order or the grammer, it focuses solely on counting how many times a word appears in the input (text, sentence, document etc.).

The bag of words model can be used for a different amount of things, 
like text classification, sentiment analysis and clustering (GeeksforGeeks, 2025).

For Bag Of Words to work there are some key components.

**Vocabulary** : Reading through a file and saving each unique word into a vocabulary.

**Document as a Vector** : Each input file wll be represented as a vector. Each element show the frequency of the words from the vocabulary in that
input file. The frequency of each word is used as a feature for the model. Shown in table 1 is an example of how this could look like.

Example input: "Hello World, I love pizza"

| Word       | Count (Frequency) |
|------------|-------------------|
| Hello      | 1                 |
| World      | 1                 |
| I          | 1                 |
| love       | 1                 |
| Hamburgers | 0                 |
| Pizza      | 1                 |

_Table 1: Input file vector example_

An important step is **Pre-processing**, this means to process the text for unused characters in this case. 
This means removing reading characters like questions mark and other non-word characters, Removing extra spaces and
Converting the text to lowercase. 

The next step is **Counting words**, here we make the table 1 example. This can be stored in a dataframe or a dictionary.
The basis of this function comes down to: If it does not exist add word to the dictionary. If it does exist add 1 to the count of the word.

The Last step is to select the vocabulary size, to manage the memory and computational complexity we limit the vocabulary to the top N (user chosen) most frequent words.
These top N words are from across all input files (if more then one were given). This reduced vocabulary then forms the final feature set.

Now u have the Top N most important words, these can be used for various machine learning related trainings.

**Why use bag of words?**

The model is simple, easily implementable and can be used for various natural language processing tasks. The results
can be visualised making them easy to interpret as well.

However, it comes of the cost of loosing context, and working wit large data sets can give efficiency issues.


### Embedding

## How to use this package

# References

- GeeksforGeeks. (2025a, juli 11). NLP | Wordlist Corpus. GeeksforGeeks. https://www.geeksforgeeks.org/nlp/nlp-wordlist-corpus/
- GeeksforGeeks. (2025, 17 juli). Bag of words (BoW) model in NLP. GeeksforGeeks. https://www.geeksforgeeks.org/nlp/bag-of-words-bow-model-in-nlp/
