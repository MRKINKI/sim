import json
from .tool.ngram.ngram_transform import NgramTransform
from .tfidf.build_tfidf import BuildTfidf
from functools import partial
from .tool import tokenize
from .topic.build_lda import BuildLDA
import pickle
from .embedding.fasttext import BuildFastText
from .embedding.w2v import BuildWord2Vec
from .classification.vocab import DataVocabs
import numpy as np
import collections
import os
from .tfidf.ranker import TfidfRanker
import pymongo


class Dataset:
    def __init__(self, conf):
        self.ngram_transform = NgramTransform()
        self.buid_tfidf = BuildTfidf()
        self.build_lda = BuildLDA()
        self.build_fasttext = BuildFastText()
        self.build_word2vec = BuildWord2Vec()

        self.tokenizer = tokenize.get_class('corenlp')()

        self.raw_file = conf.get('path', 'raw_file')
        self.segment_file = conf.get('path', 'segment_file')
        self.ngram_file = conf.get('path', 'ngram_file')
        self.train_file = conf.get('path', 'train_file')
        self.test_file = conf.get('path', 'test_file')
        self.vocab_file = conf.get('path', 'vocab_file')

        self.fast_text_model_file = conf.get('path', 'fast_text_model_file')
        self.lda_metadata_file = conf.get('path', 'lda_metadata_file')
        self.word2vec_model_file = conf.get('path', 'word2vec_model_file')
        self.tfidf_metadata_path = conf.get('path', 'tfidf_metadata_path')

        self.ngram_num = conf.getint('model', 'ngram_num')
        self.train_data_rate = conf.getfloat('model', 'train_data_rate')

        self.prepro_fields = json.loads(conf.get('key', 'prepro_field'))
        self.all_fields = json.loads(conf.get('key', 'all_field'))
        self.tfidf_fields = json.loads(conf.get('key', 'tfidf_field'))
        self.lda_fields = json.loads(conf.get('key', 'lda_field'))

        self.id_key = conf.get('key', 'id')

        self.data_set = []
        self.vocab = None

        self.msj_smi_collection = pymongo.MongoClient('192.168.1.145', 27017).food['msj_smi']

    @staticmethod
    def prepro_data(input_file, output_file, fields, fun):
        with open(output_file, 'w', encoding='utf-8') as fout:
            with open(input_file, encoding='utf-8') as fin:
                for line in fin:
                    sample = json.loads(line.strip())
                    for field in fields:
                        sample[field] = fun(sample[field])
                    fout.write(json.dumps(sample) + '\n')

    def segment_data(self, input_file, output_file, fields, character):
        get_segment = partial(self.tokenizer.segment, character=character)
        self.prepro_data(input_file, output_file, fields, get_segment)

    def ngram_data(self, input_file, output_file, fields):
        get_ngram = partial(self.ngram_transform.get_ngrams, n=self.ngram_num)
        self.prepro_data(input_file, output_file, fields, get_ngram)

    @staticmethod
    def save_data(samples, file):
        with open(file, 'w', encoding='utf-8') as fout:
            for sample in samples:
                fout.write(json.dumps(sample)+'\n')

    def train_test_split(self):
        np.random.shuffle(self.data_set)
        train_num = int(len(self.data_set)*self.train_data_rate)
        self.save_data(self.data_set[:train_num], self.train_file)
        self.save_data(self.data_set[train_num:], self.test_file)

    def build_vocab(self, src_fields, tgt_fields, mc=50):
        data_vocab = DataVocabs()
        with open(self.train_file, encoding='utf-8') as fin:
            for line in fin:
                sample = json.loads(line.strip())

                for tgt_field in tgt_fields:
                    data_vocab.get_tgt_vocab(tgt_field)
                    tgts = sample[tgt_field]
                    if isinstance(tgts, list):
                        for tgt in tgts:
                            data_vocab.tgt_vocab_dict[tgt_field].add(tgt)
                    else:
                        data_vocab.tgt_vocab_dict[tgt_field].add(tgts)

                for src_field in src_fields:
                    for gram in sample[src_field]:
                        feature = src_field + '@' + gram
                        data_vocab.src_vocab.add(feature)
        data_vocab.src_vocab.filter_tokens_by_cnt(min_cnt=mc)
        print(data_vocab.src_vocab.size())
        pickle.dump(data_vocab, open(self.vocab_file, 'wb'))

    def load_data(self, cat='ngram'):
        file = self.ngram_file
        if cat == 'segment':
            file = self.segment_file
        print(file)
        with open(file, encoding='utf-8') as fin:
            for line in fin:
                sample = json.loads(line.strip())
                self.data_set.append(sample)

    def load_vocab(self):
        self.vocab = pickle.load(open(self.vocab_file, 'rb'))

    def transform(self, input_file, output_file, src_fields, tgt_fields, predict=False):
        X, Y = [], collections.defaultdict(list)
        with open(input_file, encoding='utf-8') as fin:
            for line in fin:
                sample = json.loads(line.strip())
                if not predict:
                    for tgt_field in tgt_fields:
                        tgts = sample[tgt_field]
                        # print(tgt_field, tgts)
                        if isinstance(tgts, list):
                            Y[tgt_field].append(self.vocab.tgt_vocab_dict[tgt_field].convert_to_ids(tgts))
                        else:
                            Y[tgt_field].append(self.vocab.tgt_vocab_dict[tgt_field].get_id(tgts))

                x = []
                for src_field in src_fields:
                    features = [src_field + '@' + gram for gram in sample[src_field]]
                    x.extend(self.vocab.src_vocab.convert_to_ids(features))
                X.append(x)
        pickle.dump({'x': X,
                     'y': Y,
                     'src_size': self.vocab.src_vocab.size()}, open(output_file, 'wb'))

    def build_tfidf_model(self):
        for field in self.tfidf_fields:
            metadata = self.buid_tfidf.build(self.data_set,
                                             self.id_key,
                                             field)
            pickle.dump(metadata, open(os.path.join(self.tfidf_metadata_path, field), 'wb'))
            del metadata

    def build_lda_model(self):
        lda_field_metadata = {}
        for field in self.lda_fields:
            metadata = self.build_lda.build(self.data_set,
                                            self.id_key,
                                            field)
            lda_field_metadata[field] = metadata
        pickle.dump(lda_field_metadata, open(self.lda_metadata_file, 'wb'))

    def build_fasttext_model(self):
        model = self.build_fasttext.train(self.data_set,
                                          'illustration')
        pickle.dump(model, open(self.fast_text_model_file, 'wb'))

    def build_word2vec_model(self):
        model = self.build_word2vec.train(self.data_set,
                                          'illustration')
        pickle.dump(model, open(self.word2vec_model_file, 'wb'))

    def build_smi_data(self):
        smi_dict = collections.defaultdict(dict)
        for field in self.tfidf_fields:
            print(field)
            model_file = os.path.join(self.tfidf_metadata_path, field)
            tfidf_ranker = TfidfRanker(model_path=model_file)
            term_dict = tfidf_ranker.rank_all()
            for term_id, value in term_dict.items():
                smi_dict[term_id][field] = value
        smi_data = []
        for term_id, values in smi_dict.items():
            sample = {'id': term_id}
            for field in values:
                sample[field] = values[field]
            smi_data.append(sample)
        self.msj_smi_collection.insert_many(smi_data)
