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


class Dataset:
    def __init__(self, conf):
        self.ngram_transform = NgramTransform()
        self.buid_tfidf = BuildTfidf()
        self.build_lda = BuildLDA()
        self.build_fasttext = BuildFastText()
        self.build_word2vec = BuildWord2Vec()

        self.tokenizer = tokenize.get_class('corenlp')()

        self.tfidf_metadata_file = conf.get('path', 'tfidf_metadata_file')
        self.raw_file = conf.get('path', 'raw_file')
        self.segment_file = conf.get('path', 'segment_file')
        self.ngram_file = conf.get('path', 'ngram_file')
        self.train_file = conf.get('path', 'train_file')
        self.test_file = conf.get('path', 'test_file')
        self.vocab_file = conf.get('path', 'vocab_file')

        self.fast_text_model_file = conf.get('path', 'fast_text_model_file')
        self.lda_metadata_file = conf.get('path', 'lda_metadata_file')
        self.word2vec_model_file = conf.get('path', 'word2vec_model_file')

        self.ngram_num = conf.getint('model', 'ngram_num')
        self.train_data_rate = conf.getfloat('model', 'train_data_rate')

        self.prepro_fields = json.loads(conf.get('key', 'prepro_field'))
        self.all_fields = json.loads(conf.get('key', 'all_field'))
        self.lda_fields = json.loads(conf.get('key', 'lda_field'))

        self.id_key = conf.get('key', 'id')

        self.data_set = []
        self.vocab = None

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

    def build_vocab(self, tgt_fields):
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

                for key, value in sample.items():
                    if key not in tgt_fields:
                        for gram in value:
                            feature = key + '@' + gram
                            data_vocab.src_vocab.add(feature)
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

    def transform(self, input_file, output_file, tgt_fields, predict=False):
        X, Y = [], collections.defaultdict(list)
        with open(input_file, encoding='utf-8') as fin:
            for line in fin:
                sample = json.loads(line.strip())
                if not predict:
                    for tgt_field in tgt_fields:
                        tgts = sample[tgt_field]
                        if isinstance(tgts, list):
                            Y[tgt_field].append(self.vocab.tgt_vocab_dict[tgt_field].convert_to_ids(tgts))
                        else:
                            Y[tgt_field].append(self.vocab.tgt_vocab_dict[tgt_field].get_id(tgts))

                x = []
                for key, value in sample.items():
                    if key not in tgt_fields:
                        features = [key + '@' + gram for gram in value]
                        x.extend(self.vocab.src_vocab.convert_to_ids(features))
                X.append(x)
        pickle.dump({'x': X,
                     'y': Y,
                     'src_size': self.vocab.src_vocab.size()}, open(output_file, 'wb'))

    def build_tfidf_model(self):
        all_field_metadata = {}
        for field in self.all_fields:
            metadata = self.buid_tfidf.build(self.data_set,
                                             self.id_key,
                                             field)
            all_field_metadata[field] = metadata
        pickle.dump(all_field_metadata, open(self.tfidf_metadata_file, 'wb'))

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
