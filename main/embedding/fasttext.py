import os
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText


class BuildFastText:
    def __init__(self):
        pass

    def get_texts(self, corpus, field):
        texts = []
        for idx, sample in enumerate(corpus):
            text_words = sample[field]
            texts.append(text_words)
        return texts

    def train(self, corpus, field):
        texts = self.get_texts(corpus, field)
        model = FastText(texts, iter=100)
        return model
