import scipy.sparse as sp
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.linear_model import LogisticRegression
from .utils import get_count_matrix, binarize_label


class OVR:
    def __init__(self):
        pass

    def train(self, train_data_file, test_data_file, metadata_file):
        raw_train = pickle.load(open(train_data_file, 'rb'))
        raw_test = pickle.load(open(test_data_file, 'rb'))
        train_x_count_matrix = get_count_matrix(raw_train['x'], raw_train['src_size'])
        test_x_count_matrix = get_count_matrix(raw_test['x'], raw_train['src_size'])

        tfidf_transformer = TfidfTransformer()
        model = OneVsRestClassifier(LogisticRegression(C=10, class_weight='balanced'))

        tfidf_transformer.fit(train_x_count_matrix)

        train_x = tfidf_transformer.transform(train_x_count_matrix)
        test_x = tfidf_transformer.transform(test_x_count_matrix)
        train_y = binarize_label(raw_train['y'], raw_train['tgt_size'])
        test_y = binarize_label(raw_test['y'], raw_train['tgt_size'])
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        metadata = {'tfidf': tfidf_transformer,
                    'model': model}
        pickle.dump(metadata, open(metadata_file, 'wb'))
        self.evaluate(test_y, pred_y)

    @staticmethod
    def evaluate(test_y, pred_y):
        precitions, recalls, f1s = [], [], []
        for true, pred in zip(test_y, pred_y):
            true_ids = [i for i in range(len(true)) if true[i] > 0]
            pred_ids = [i for i in range(len(pred)) if pred[i] > 0]
            if len(pred_ids) > 0:
                precition = len(set(true_ids) & set(pred_ids)) / len(pred_ids)
            else:
                precition = 0
            recall = len(set(true_ids) & set(pred_ids)) / len(true_ids)
            if precition + recall == 0:
                f1 = 0
            else:
                f1 = 2*precition*recall / (precition+recall)
            precitions.append(precition)
            recalls.append(recall)
            f1s.append(f1)
        print('precision:%f' % np.average(precitions))
        print('recall:%f' % np.average(recalls))
        print('f1:%f' % np.average(f1s))
