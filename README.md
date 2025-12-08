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





# References

- GeeksforGeeks. (2025a, juli 11). NLP | Wordlist Corpus. GeeksforGeeks. https://www.geeksforgeeks.org/nlp/nlp-wordlist-corpus/
- 