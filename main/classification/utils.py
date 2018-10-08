import scipy.sparse as sp
from collections import Counter
import numpy as np


def get_count_matrix(features, size):
    row, col, data = [], [], []
    for idx, sample in enumerate(features):
        counts = Counter(sample)
        col.extend(counts.keys())
        row.extend([idx] * len(counts))
        data.extend(counts.values())
    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(idx + 1, size)
    )
    count_matrix.sum_duplicates()
    return count_matrix


def get_id(binary_tgt, alpha):
    ids = []
    for bt in binary_tgt:
        ids.append([idx for idx in range(len(bt)) if bt[idx] > alpha])
    return ids


def binarize_label(tgts, tgt_size):
    binarize_label_list = []
    for i in range(len(tgts)):
        temp = [0 for j in range(tgt_size)]
        for tgt in tgts[i]:
            temp[tgt] = 1
        binarize_label_list.append(temp)
    return np.array(binarize_label_list)
