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
from sklearn import svm
import collections
import xgboost as xgb
from sklearn.feature_selection import chi2
import math


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
        choose_feature_list = []
        train_chain_pred_history = sp.csr_matrix([])
        feature_num = 20000
        model_type = 'svm'
        for i in range(classification_model_num):
            chains.append(self.get_model(model_type))

        chain_train_x = train_x
        print(chain_train_x.shape)
        for idx, tgt_field in enumerate(chain_tgt_fields):
            print(tgt_field)
            # chain.fit(chain_train_x, raw_train['y'][tgt_field])
            # train_pred = chain.predict(chain_train_x)
            # train_pred_pro = sp.csr_matrix(chain.predict_proba(chain_train_x))

            choose_feature = self.feature_slect(train_x, raw_train['y'][tgt_field], feature_num)
            if train_chain_pred_history.shape[1] != 0:
                chain_train_x = sp.hstack([train_x[:, choose_feature], train_chain_pred_history])
            else:
                chain_train_x = train_x[:, choose_feature]
            choose_feature_list.append(choose_feature)
            print(chain_train_x.shape)
            chains[idx], train_pred_pro = self.get_fit(chains[idx],
                                                       model_type,
                                                       chain_train_x,
                                                       raw_train['y'][tgt_field],
                                                       )
            if train_chain_pred_history.shape[1] != 0:
                train_chain_pred_history = sp.hstack([train_chain_pred_history, train_pred_pro])
            else:
                train_chain_pred_history = train_pred_pro
            # chain_train_x = sp.hstack([chain_train_x, train_pred_pro])
            # chain_train_x = sp.csr_matrix(chain_train_x)
            # print(train_pred)
        self.chains = chains
        self.chain_tgt_fields = chain_tgt_fields
        metadata = {'tfidf': tfidf_transformer,
                    'model': self}
        pickle.dump(metadata, open(metadata_file, 'wb'))

        chain_test_x = test_x
        test_chain_pred_history = sp.csr_matrix([])
        for chain, tgt_field, choose_feature in zip(chains, chain_tgt_fields, choose_feature_list):
            # preds = chain.predict(chain_test_x)
            # test_pred_pro = sp.csr_matrix(chain.predict_proba(chain_test_x))
            if test_chain_pred_history.shape[1] != 0:
                chain_test_x = sp.hstack([test_x[:, choose_feature], test_chain_pred_history])
            else:
                chain_test_x = test_x[:, choose_feature]
            test_pred_pro, test_pred = self.get_predict(chain, model_type, chain_test_x)

            if test_chain_pred_history.shape[1] != 0:
                test_chain_pred_history = sp.hstack([test_chain_pred_history, test_pred_pro])
            else:
                test_chain_pred_history = test_pred_pro
            # chain_test_x = sp.hstack([chain_test_x, test_pred_pro])
            # chain_test_x = sp.csr_matrix(chain_test_x)

            print(tgt_field)
            print(metrics.classification_report(raw_test['y'][tgt_field], test_pred, digits=4))

    @staticmethod
    def get_model(model_type):
        if model_type == 'lr':
            return LogisticRegression(C=10,
                                      # solver='lbfgs',
                                      solver='liblinear',
                                      class_weight='balanced',
                                      # multi_class='multinomial',
                                      # multi_class='ovr',
                                      max_iter=1000,
                                      n_jobs=-1)
        elif model_type == 'svm':
            return svm.SVC(C=10,
                           kernel='linear',
                           class_weight='balanced',
                           probability=True,
                           )
        elif model_type == 'xgb':
            return 'xgb'

    @staticmethod
    def get_predict(model, model_type, test_x):
        test_pred_pro, test_pred = None, None
        if model_type in ['lr', 'svm']:
            test_pred_pro = model.predict_proba(test_x)
            test_pred = np.argmax(test_pred_pro, 1)
        elif model_type in ['xgb']:
            data_test = xgb.DMatrix(test_x, [0]*test_x.shape[0])
            test_pred_pro = model.predict(data_test)
            test_pred = np.argmax(test_pred_pro, 1)
        return sp.csr_matrix(test_pred_pro), test_pred

    @staticmethod
    def get_fit(model, model_type, train_x, train_y):
        if model_type in ['lr', 'svm']:
            model.fit(train_x, train_y)
            train_pred_pro = sp.csr_matrix(model.predict_proba(train_x))
            return model, train_pred_pro

        elif model_type in ['xgb']:
            num_class = len(collections.Counter(train_y))
            params = {'max_depth': 6,
                      'eta': 0.1,
                      # 'objective':'multi:softmax',
                      'objective': 'multi:softprob',
                      'num_class': num_class,
                      'nthread': 5,
                      }
            train_rate = 0.95
            sub_train_num = int(train_x.shape[0] * train_rate)
            sub_train_x, sub_train_y = train_x[:sub_train_num], train_y[:sub_train_num]
            eval_x, eval_y = train_x[sub_train_num:], train_y[sub_train_num:]
            data_train = xgb.DMatrix(sub_train_x, sub_train_y)
            data_test = xgb.DMatrix(eval_x, eval_y)
            watchlist = [(data_test, 'eval'), (data_train, 'train')]
            n_round = 10000
            bst = xgb.train(params, data_train, num_boost_round=n_round, evals=watchlist, early_stopping_rounds=50)
            train_pred_pro = sp.csr_matrix(bst.predict(data_train))
            return bst, train_pred_pro

    @staticmethod
    def feature_slect(count_matrix, tgt, feature_num):
        chi, pval = chi2(count_matrix, tgt)
        feature_dict = {i: v for i, v in enumerate(chi) if not math.isnan(v)}
        sorted_feature_dict = sorted(feature_dict.items(), key=lambda f: f[1], reverse=True)
        choose_features = [t[0] for t in sorted_feature_dict][:feature_num]
        return choose_features
