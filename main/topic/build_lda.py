from gensim import corpora, models, similarities


class BuildLDA:
    def __init__(self):
        pass

    def get_texts(self, corpus, id_key, field):
        texts = []
        idx2id = {}
        for idx, sample in enumerate(corpus):
            idx2id[idx] = sample[id_key]
            text_words = sample[field]
            texts.append(text_words)
        return texts, idx2id

    def build(self, corpus, id_key, field):
        texts, idx2id = self.get_texts(corpus, id_key, field)
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=5, no_above=0.8)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        lda = models.LdaModel(corpus_tfidf, id2word=dictionary, passes=5, num_topics=20, iterations=500)
        index = similarities.MatrixSimilarity(lda[corpus_tfidf])
        metadata = {
            'idx2id': idx2id,
            'id2idx': {text_id: idx for idx, text_id in idx2id.items()},
            'lda_model': lda,
            'tfidf': tfidf,
            'dictionary': dictionary,
            'lda_index': index,
        }
        return metadata
