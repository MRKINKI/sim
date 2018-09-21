# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from collections import Counter
import pickle
import math
from .utils import mhash


class BuildTfidf:
    def __init__(self):
        self.hash_size = int(math.pow(2, 24))
        self.idx2id = {}

    def get_count_matrix(self, corpus, id_key, text_key):
        row, col, data = [], [], []
        for idx, sample in enumerate(corpus):
            self.idx2id[idx] = sample[id_key]
            text_ngrams = sample[text_key]
            counts = Counter([mhash(gram, self.hash_size) for gram in text_ngrams])
            col.extend(counts.keys())
            row.extend([idx] * len(counts))
            data.extend(counts.values())
            if idx % 1000 == 0:
                print('complete %d' % idx)
        count_matrix = sp.csr_matrix(
            (data, (row, col)), shape=(idx + 1, self.hash_size)
        )
        count_matrix.sum_duplicates()
        return count_matrix

    def get_tfidf_matrix(self, cnts):
        """
        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        Ns = self.get_doc_freqs(cnts)
        idfs = np.log((cnts.shape[0] - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0
        idfs = sp.diags(idfs, 0)
        tfs = cnts.log1p()
        tfidfs = tfs.dot(idfs)
        return tfidfs, Ns

    @staticmethod
    def get_doc_freqs(cnts):
        binary = (cnts > 0).astype(int)
        freqs = np.array(binary.sum(0)).squeeze()
        return freqs

    @staticmethod
    def matrix_norm(tfidf_matrix):
        norm = 1.0 / np.sqrt(np.array(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).squeeze())
        norm_matrix = sp.diags(norm, 0)
        norm_tfidf_matrix = norm_matrix.dot(tfidf_matrix)
        return norm_tfidf_matrix

    def build(self, corpus, metadata_file, id_key, text_key):
        count_matrix = self.get_count_matrix(corpus, id_key, text_key)
        tfidf_matrix, freqs = self.get_tfidf_matrix(count_matrix)
        tfidf_matrix = self.matrix_norm(tfidf_matrix)
        metadata = {
            'idx2id': self.idx2id,
            'freqs': freqs,
            'tfidf_matrix': tfidf_matrix,
            'hash_size': self.hash_size,
        }
        pickle.dump(metadata, open(metadata_file, 'wb'))
