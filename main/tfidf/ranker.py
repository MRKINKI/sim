# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
from multiprocessing.pool import ThreadPool
from functools import partial
import pickle
from .utils import mhash


class TfidfRanker(object):

    def __init__(self, model_path=None, strict=True):
        self.model = pickle.load(open(model_path, 'rb'))
        self.strict = strict
        
    def idx2id(self, idx):
        return self.model['idx2id'][idx]

    def rank(self, words, field, k=1):
        spvec = self.text2spvec(words, field)
        res = spvec * self.model[field]['tfidf_matrix'].T
        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]
        scores = res.data[o_sort]
        indexs = [i for i in res.indices[o_sort]]
        return zip(indexs, scores)

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

    def get_rank_id_score(self, idxes):
        spvecs = self.model['tfidf_matrix'][idxes, :]
        scores = spvecs.dot(self.model['tfidf_matrix'].T)
        scores = scores.toarray().squeeze()
        return scores, np.argsort(-scores)

    def combine_score_id(self, scores, ranks, topk):
        score_id = []
        for idx in ranks[:topk]:
            text_id = self.idx2id(idx)
            score = scores[idx]
            score_id.append((text_id, score))
        return score_id

    def rank_all(self, topk=5):
        batch_size = 1000
        term_dict = {}
        for i in range(0, self.model['tfidf_matrix'].shape[0], batch_size):
            idxes = list(range(i, min(i+batch_size, self.model['tfidf_matrix'].shape[0])))
            # print(idxes)
            scores, ranks = self.get_rank_id_score(idxes)
            for idx in range(len(idxes)):
                term_id = self.idx2id(i+idx)
                term_dict[term_id] = self.combine_score_id(scores[idx], ranks[idx], topk)
            print(i)
        return term_dict

    def rank_from_id(self, text_id, topk=5):
        cands = []
        idx = self.model['id2idx'][text_id]
        # scores = self.model['sim_matrix'][idx].toarray().squeeze()
        # ranks = np.argsort(-scores)
        scores, ranks = self.get_rank_id_score([idx])
        for idx in ranks[:topk]:
            text_id = self.idx2id(idx)
            score = scores[idx]
            # score = min([score, 1.0])
            cands.append({'id': text_id,
                          'score': score})
        return cands

    @staticmethod
    def norm(data):
        return data / np.sqrt(sum(data * data))

    def text2spvec(self, words, field):
        data_num = len(self.model[field]['idx2id'])
        wids = [mhash(w, self.model['hash_size']) for w in words]
        if len(wids) == 0:
            raise RuntimeError('No valid word in: %s' % words)
        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)
        # Count IDF
        Ns = self.model[field]['freqs'][wids_unique]
        idfs = np.log((data_num - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0
        # TF-IDF
        data = np.multiply(tfs, idfs)
        data = self.norm(data)
        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])
        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.model['field']['hash_size'])
        )
        return spvec
