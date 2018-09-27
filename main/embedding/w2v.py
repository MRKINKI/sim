from gensim.models import Word2Vec


class BuildWord2Vec:
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
        model = Word2Vec(texts, iter=20)
        return model
