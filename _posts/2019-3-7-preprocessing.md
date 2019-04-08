---
published: true
title: Preprocessing
collection: articles
layout: single
author_profile: false
read_time: true
categories: [articles]
excerpt : "Natural Language Processing"
header :
    overlay_image: "https://raphaellederman.github.io/assets/images/night.jpg"
    teaser_image: "https://raphaellederman.github.io/assets/images/night.jpg"
toc: true
toc_sticky: true
---

In this article, we will describe some of the main elements that are used in the preprocessing of text data, and provide a short implementation in Python. The preprocessing is the very first step in building a proper natural language processing pipeline, and is crucial in order to obtain a standardized data that can be used with machine learning models.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Overview 

The preprocessing is the first step of a NLP pipeline : this is where we convert raw text document to cleaned lists of words. In order to complete this process, it is firt important to perform text data noise removal using regular expressions. This task can include removing headers, footers, deleting unwanted characters or simply replacing contractions with their expansions in order to avoid a messy $$\textit{tokenization}$$ of the corpus, which is the next step. 

Tokenization consists in splitting large chunks of tesxt into sentences, and sentences into a list of single words, also called $$\textit{tokens}$$. This step, also referred to as $$\textit{segmentation}$$ or $$\textit{lexical analysis}$$, is necessary in order to perform further processing.

Once tokens are obtained, it is common to standardize them. Standardization for instance includes lowercasing tokens or deleting some punctuation characters that are not crucial to the understanding of the text. The removing of stopwords in order to retain only words with meaning is also an important standardization step as it allows to get rid of words that are too common like 'a', 'the' or 'an'. One way to see this task is to see it as a way to put all words on an equal footing.

Then, there are methods available in order to replace words by their grammatical $$\textit{root}$$ : the goal of both $$\textit{stemming}$$ and $$\textit{lemmatization}$$ is to reduce derivationally related forms of a word to a common base form. Stemming eliminates affixes (suffixes, prefixes, infixes, circumfixes) from a word in order to obtain a word stem while lemmatization is able to capture canonical forms based on a word's lemma. Families of derivationally related words with similar meanings, such as 'am', 'are', 'is' would then be replace by the word 'be'. 

Finally, in the context of word sense disambiguation, part-of-speech tagging is used in order to mark up words in a corpus as corresponding to a particular part of speech, based on both its definition and its context. This can be used to improve the accuracy of the lemmatization process, or just to have a better understanding of the $$\textit{meaning}$$ of a sentence.

After all these preprocessibng steps, we need to convert the text files into numerical feature vectors. This step, called $$\textit{vectorization}$$, consists in converting a corpus of documents into a matrix of token counts, the number of features being equal to the vocabulary size. This conversion allows to perform visualization (word frequencies, word clouds...) or classification for instance. Of course, other ways of representing the text data into a numerical feature vectors exist. For instance, in order to use deep learning models, it is common to use a richer representation, such as word embeddings (but this will be the subject of our next short article).

## Short implementation 

The implementation of a preprocessing program can be done using the following functions (it is just one basic way of preprocessing text data, many other options exist), along with the NLTK package (Natural Language ToolKit, one of the best-known NLP libraries in the Python ecosystem) and some Keras preprocessing tools :

```python
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords as sw, wordnet as wn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def preprocess(document, max_features, max_sentence_len):
    """
    Returns a normalized, lemmatized list of tokens from a document by
    applying segmentation (breaking into sentences), then word/punctuation
    tokenization, and finally part of speech tagging. It uses the part of
    speech tags to look up the lemma in WordNet, and returns the lowercase
    version of all the words, removing stopwords and punctuation.
    """
    lemmatized_tokens = []

    # Clean the text using a few regular expressions
    document = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", document)
    document = re.sub(r"what's", "what is ", document)
    document = re.sub(r"\'s", " ", document)
    document = re.sub(r"\'ve", " have ", document)
    document = re.sub(r"can't", "cannot ", document)
    document = re.sub(r"n't", " not ", document)
    document = re.sub(r"i'm", "i am ", document)
    document = re.sub(r"\'re", " are ", document)
    document = re.sub(r"\'d", " would ", document)
    document = re.sub(r"\'ll", " will ", document)
    document = re.sub(r"(\d+)(k)", r"\g<1>000", document)

    # Break the document into sentences
    for sent in sent_tokenize(document):

        # Break the sentence into part of speech tagged tokens
        for token, tag in pos_tag(wordpunct_tokenize(sent)):

            # Apply preprocessing to the tokens
            token = token.lower()
            token = token.strip()
            token = token.strip('_')
            token = token.strip('*')

            # If punctuation or stopword, ignore token and continue
            if token in set(sw.words('english')) or all(char in set(string.punctuation) for char in token):
                continue

            # Lemmatize the token
            lemma = lemmatize(token, tag)
            lemmatized_tokens.append(lemma)

    cleaned_document = ' '.join(lemmatized_tokens)
    vectorized_document, word_index = vectorize(np.array(cleaned_document, max_features, max_sentence_len)[np.newaxis])
    return vectorized_document, word_index

def lemmatize(token, tag):
	"""
	Converts the tag to a WordNet POS tag, then uses that
	tag to perform an accurate WordNet lemmatization.
	"""
	tag = {
	    'N': wn.NOUN,
	    'V': wn.VERB,
	    'R': wn.ADV,
	    'J': wn.ADJ
	}.get(tag[0], wn.NOUN)

	return WordNetLemmatizer().lemmatize(token, tag)

def vectorize(doc, max_features, max_sentence_len):
	"""
	Converts a document into a sequence of indices of length max_sentence_len retaining only max_features unique words
	"""
	tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(doc)
    doc = tokenizer.texts_to_sequences(doc)
    doc_pad = pad_sequences(doc, padding = 'pre', truncating = 'pre', maxlen = max_sentence_len)
    return np.squeeze(doc_pad), tokenizer.word_index
```

> **Conclusion** : the preprocessing of raw text data is indispensable in order to build an efficient natural language processing pipeline. As we have said in this short article, the main steps are the following : noise removal, tokenization, normalization, stemming/lemmatization, part-of-speech tagging and vectorization.
