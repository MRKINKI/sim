import pickle
import numpy as np


class TopicRanker:
    def __init__(self, model_path=None):
        self.model = pickle.load(open(model_path, 'rb'))

    def idx2id(self, idx, field):
        return self.model[field]['idx2id'][idx]

    def rank(self, words, field, k=1):
        lda_index = self.model[field]['lda_index']
        spvec = self.text2spvec(words, field)
        sims = lda_index[spvec].squeeze()
        sorted_sims = sorted(enumerate(sims.squeeze()), key=lambda f: -f[1])
        return sorted_sims[:k]

    def batch_closest_patterns(self, queries, k=1, num_workers=None):
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_patterns, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def rank_from_text(self, words, field, topk=5):
        cands = []
        ranks = self.rank(words, field, topk)
        for idx, score in ranks:
            text_id = self.idx2id(idx, field)
            # score = min([score, 1.0])
            cands.append({'id': text_id,
                          'score': score})
        return cands

    def rank_from_id(self, text_id, field, topk=5):
        cands = []
        idx = self.model[field]['id2idx'][text_id]
        scores = self.model[field]['sim_matrix'][idx].toarray().squeeze()
        ranks = np.argsort(-scores)
        for idx in ranks[:topk]:
            text_id = self.idx2id(idx, field)
            score = scores[idx]
            # score = min([score, 1.0])
            cands.append({'id': text_id,
                          'score': score})
        return cands

    @staticmethod
    def norm(data):
        return data / np.sqrt(sum(data * data))

    def text2spvec(self, words, field):
        dictionary = self.model['illustration']['dictionary']
        query_bow = [dictionary.doc2bow(words)]
        tfidf_model = self.model[field]['tfidf']
        corpus_tfidf = tfidf_model[query_bow]

        lda_model = self.model[field]['lda_model']

        spvec = lda_model[corpus_tfidf]
        return spvec