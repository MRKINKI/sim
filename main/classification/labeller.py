from . import ovr
import pickle
from .utils import get_count_matrix, get_id
import json
import pandas as pd


class Labbeller:
    def __init__(self):
        pass

    def run(self, train_data_file, test_data_file):
        pass

    def predict(self, raw_file, data_file, meta_file, vocab):
        metadata = pickle.load(open(meta_file, 'rb'))
        model = metadata['model']
        tfidf_transformer = metadata['tfidf']

        data = pickle.load(open(data_file, 'rb'))

        count_matrix = get_count_matrix(data['x'], data['src_size'])
        tfidf = tfidf_transformer.transform(count_matrix)

        # pred_y = model.predict_proba(tfidf)
        # ids = get_id(pred_y, alpha=0.3)

        all_ids = model.predict(tfidf)
        labels = []
        chain_tgt_fields = model.chain_tgt_fields
        for ids in all_ids:
            label = []
            for i, field in enumerate(chain_tgt_fields):
                label.append(vocab.tgt_vocab_dict[field].get_token(ids[i]))
            labels.append(label)

        result = []
        with open(raw_file, encoding='utf-8') as fin:
            for idx, line in enumerate(fin):
                sample = json.loads(line.strip())
                sample['tag'] = labels[idx]
                result.append(sample)
        return pd.DataFrame(result)
