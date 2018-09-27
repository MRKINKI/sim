from gensim import corpora, models, similarities
import numpy as np
import scipy.sparse as sp


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

    def tranform_sparse(self, raw_data, topic_num):
        row, col, data = [], [], []
        for idx, sample in enumerate(raw_data):
            col.extend([t[0] for t in sample])
            row.extend([idx] * len(sample))
            data.extend([t[1] for t in sample])
        count_matrix = sp.csr_matrix(
            (data, (row, col)), shape=(idx + 1, topic_num)
        )
        count_matrix.sum_duplicates()
        return count_matrix

    def build(self, corpus, id_key, field):
        topic_num = 50
        texts, idx2id = self.get_texts(corpus, id_key, field)
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=5, no_above=0.8)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        # corpus = tfidf[corpus]

        lda = models.LdaModel(corpus, id2word=dictionary, passes=10, num_topics=topic_num, iterations=500)
        doc_topic = lda[corpus]
        index = similarities.MatrixSimilarity(doc_topic)
        doc_topic_matrix_sparse = self.tranform_sparse(doc_topic, topic_num)
        metadata = {
            'idx2id': idx2id,
            'id2idx': {text_id: idx for idx, text_id in idx2id.items()},
            'lda_model': lda,
            'tfidf': tfidf,
            'dictionary': dictionary,
            'lda_index': index,
            'doc_topic_matrix': doc_topic_matrix_sparse,
            'term_topic_matrix': sp.csr_matrix(lda.get_topics().T),
        }
        return metadata
