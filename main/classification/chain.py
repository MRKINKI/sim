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
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class Chain:
    def __init__(self):
        self.chains = None
        self.chain_tgt_fields = None

    def train(self, train_data_file, test_data_file, chain_tgt_fields, metadata_file=None):
        print('start train')
        raw_train = pickle.load(open(train_data_file, 'rb'))
        raw_test = pickle.load(open(test_data_file, 'rb'))

        print('get count matrix')
        train_x_count_matrix = get_count_matrix(raw_train['x'], raw_train['src_size'])
        test_x_count_matrix = get_count_matrix(raw_test['x'], raw_train['src_size'])
        classification_model_num = len(raw_train['y'])
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.fit(train_x_count_matrix)

        train_x = tfidf_transformer.transform(train_x_count_matrix)
        test_x = tfidf_transformer.transform(test_x_count_matrix)

        # train_y = binarize_label(raw_train['y'], raw_train['tgt_size'])
        # test_y = binarize_label(raw_test['y'], raw_train['tgt_size'])

        chains = []
        preds = []
        for i in range(classification_model_num):
            chains.append(LogisticRegression(C=10,
                                             # solver='lbfgs',
                                             solver='liblinear',
                                             class_weight='balanced',
                                             # multi_class='multinomial',
                                             # multi_class='ovr',
                                             max_iter=1000,
                                             n_jobs=-1))

        chain_train_x = train_x
        print(chain_train_x.shape)
        for chain, tgt_field in zip(chains, chain_tgt_fields):
            print(tgt_field)
            chain.fit(chain_train_x, raw_train['y'][tgt_field])
            # train_pred = chain.predict(chain_train_x)
            train_pred_pro = sp.csr_matrix(chain.predict_proba(chain_train_x))
            chain_train_x = sp.hstack([chain_train_x, train_pred_pro])
            # print(train_pred)
        self.chains = chains
        self.chain_tgt_fields = chain_tgt_fields
        metadata = {'tfidf': tfidf_transformer,
                    'model': self}
        pickle.dump(metadata, open(metadata_file, 'wb'))

        chain_test_x = test_x
        for chain, tgt_field in zip(chains, chain_tgt_fields):
            preds = chain.predict(chain_test_x)
            test_pred_pro = sp.csr_matrix(chain.predict_proba(chain_test_x))
            chain_test_x = sp.hstack([chain_test_x, test_pred_pro])
            print(tgt_field)
            print(metrics.classification_report(raw_test['y'][tgt_field], preds, digits=4))
