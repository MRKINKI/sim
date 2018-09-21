# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
from multiprocessing.pool import ThreadPool
from functools import partial
import pickle
from .utils import mhash, get_ngrams


class TfidfRanker(object):

    def __init__(self, tokenizer, model_path=None, strict=True):
        self.model = pickle.load(open(model_path, 'rb'))
        self.strict = strict
        self.tokenizer = tokenizer
        self.ngramNum = self.model['ngramNum']
        self.pattern_num = len(self.model['id2pattern'])
        
    def id2pattern(self, pattern_id):
        return self.model['id2pattern'][pattern_id]

    def closest_patterns(self, query, k=1):
        spvec = self.text2spvec(query)
        res = spvec * self.model['tfidf_matrix'].T
        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]
        pattern_scores = res.data[o_sort]
        pattern_indexs = [i for i in res.indices[o_sort]]
        return zip(pattern_indexs, pattern_scores)

    def batch_closest_patterns(self, queries, k=1, num_workers=None):
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_patterns, k=k)
            results = threads.map(closest_docs, queries)
        return results
    
    def get_candidate_patterns(self, query, max_pattern_num=5):
        cand_patterns = []
        ranks = self.closest_patterns(query, max_pattern_num)
        for pattern_id, score in ranks:
            pattern = self.id2pattern(pattern_id)
            score = min([score, 1.0])
            cand_patterns.append({'pattern': pattern,
                                  'search_score': score})
        return cand_patterns

    def parse(self, query):
        query_words = self.tokenizer.tokenize(query)
        return get_ngrams(query_words, n=self.ngramNum)

    @staticmethod
    def norm(data):
        return data / np.sqrt(sum(data * data))

    def text2spvec(self, query):
        words = self.parse(query)
        wids = [mhash(w, self.model['hash_size']) for w in words]
        if len(wids) == 0:
            raise RuntimeError('No valid word in: %s' % query)
        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)
        # Count IDF
        Ns = self.model['freqs'][wids_unique]
        idfs = np.log((self.pattern_num - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0
        # TF-IDF
        data = np.multiply(tfs, idfs)
        data = self.norm(data)
        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])
        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.model['hash_size'])
        )
        return spvec
