# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32


def save_sparse_csr(filename, matrix, metadata):
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'metadata': metadata,
    }
    np.savez(filename, **data)


def load_sparse_csr(filename):
    loader = np.load(filename)
    matrix = sp.csr_matrix((loader['data'], loader['indices'],
                            loader['indptr']), shape=loader['shape'])
    return matrix, loader['metadata'].item(0) if 'metadata' in loader else None


def mhash(token, num_buckets):
    """Unsigned 32 bit murmurhash for feature hashing."""
    return murmurhash3_32(token, positive=True) % num_buckets
