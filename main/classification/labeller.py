from . import ovr
import pickle
from .utils import get_count_matrix, get_id
import json
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pymongo


class Labbeller:
    def __init__(self):
        self.result_collection = pymongo.MongoClient('192.168.1.145', 27017).food['baidu_recipe_auto_label']

    def label(self, raw_file, data_file, meta_file, vocab):
        metadata = pickle.load(open(meta_file, 'rb'))
        model = metadata['model']
        tfidf_transformer = metadata['tfidf']

        data = pickle.load(open(data_file, 'rb'))

        count_matrix = get_count_matrix(data['x'], data['src_size'])
        tfidf = tfidf_transformer.transform(count_matrix)

        # pred_y = model.predict_proba(tfidf)
        # ids = get_id(pred_y, alpha=0.3)

        # all_ids = model.predict(tfidf)
        all_pred_pros = self.predict(model, tfidf)
        chain_tgt_fields = model.chain_tgt_fields
        labels = self.get_label(all_pred_pros, vocab, chain_tgt_fields)
        # labels = []
        # for ids in all_ids:
        #     label = []
        #     for i, field in enumerate(chain_tgt_fields):
        #         label.append(vocab.tgt_vocab_dict[field].get_token(ids[i]))
        #     labels.append(label)

        result = []
        with open(raw_file, encoding='utf-8') as fin:
            for idx, line in enumerate(fin):
                sample = json.loads(line.strip())
                sample['tag'] = labels[idx]
                result.append(sample)
        self.result_collection.insert_many(result)
        return pd.DataFrame(result)

    @staticmethod
    def predict(model, test_x):
        chain_test_x = test_x
        all_pred_pros = []
        # all_labels = []
        for chain, tgt_field in zip(model.chains, model.chain_tgt_fields):
            # preds = chain.predict(chain_test_x)
            # all_labels.append(preds)
            test_pred_pro = chain.predict_proba(chain_test_x)
            test_pred_pro_csr = sp.csr_matrix(test_pred_pro)

            all_pred_pros.append(test_pred_pro)
            chain_test_x = sp.hstack([chain_test_x, test_pred_pro_csr])
        # all_labels_concat = np.array(all_labels).T
        return all_pred_pros

    @staticmethod
    def get_label(all_pred_pros, vocab, chain_tgt_fields):
        labels = [[] for i in range(len(all_pred_pros[0]))]
        # ids = [[] for i in range(len(all_pred_pros[0]))]
        for pred_idx, tgt_field in enumerate(chain_tgt_fields):
            # print(all_pred_pros[pred_idx])
            for sample_idx, pred_pro in enumerate(all_pred_pros[pred_idx]):
                # print(pred_pro)
                for idx in np.argsort(-pred_pro):
                    label = vocab.tgt_vocab_dict[tgt_field].get_token(idx+2)
                    pro = round(pred_pro[idx], 2)
                    # labels[sample_idx].append(label)
                    # ids[sample_idx].append(idx+2)
                    if label not in ['', '<blank>', '<unk>', '其他工艺', '其它口味', '其它工艺', '其他口味']:
                        labels[sample_idx].append(label+' '+str(pro))
                        break
        return labels
