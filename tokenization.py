import numpy as np
import itertools


class Tokenizer:
    """
    From a set of text documents, create a tokenizer object to encode and decode
    """
    def __init__(self):
        self.tokens = {}

    def fit(self, documents):
        """Assign integer index to every word seen"""
        corpus = [x.split(' ') for x in documents]
        self.token_word = dict(enumerate(set(list(itertools.chain.from_iterable(corpus)))))
        self.word_token = {v: k for k, v in self.token_word.items()}
        self.vocab_size = max(self.token_word.keys()) + 1

    def int_tokenize(self, documents):
        """Get integer index for words passed"""
        documents = [x.split(' ') for x in documents]
        documents = [list(map(self.word_token.get, doc)) for doc in documents]
        return documents

    def int_detokenize(self, tokens):
        """From integer index, get word associated with token"""
        tokens = [list(map(self.token_word.get, token)) for token in tokens]
        return tokens

    def sparse_tokenize(self, tokens):
        """Transform integer inputs into one hot encoded"""
        return np.eye(self.vocab_size)[tokens]

    def sparse_detokenize(self, sparse_matrix):
        """From one hot encoded, find integer index"""
        indices = np.argmax(sparse_matrix, axis=2)
        return self.int_detokenize(indices)
