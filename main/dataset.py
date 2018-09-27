import json
from .tool.ngram.ngram_transform import NgramTransform
from .tfidf.build_tfidf import BuildTfidf
from functools import partial
from .tool import tokenize
from .topic.build_lda import BuildLDA
import pickle
from .embedding.fasttext import BuildFastText
from .embedding.w2v import BuildWord2Vec


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
        self.fast_text_model_file = conf.get('path', 'fast_text_model_file')
        self.lda_metadata_file = conf.get('path', 'lda_metadata_file')
        self.word2vec_model_file = conf.get('path', 'word2vec_model_file')

        self.ngram_num = conf.getint('model', 'ngram_num')

        self.prepro_fields = json.loads(conf.get('key', 'prepro_field'))
        self.all_fields = json.loads(conf.get('key', 'all_field'))
        self.lda_fields = json.loads(conf.get('key', 'lda_field'))

        self.id_key = conf.get('key', 'id')

        self.data_set = []

    @staticmethod
    def prepro_data(input_file, output_file, fields, fun):
        with open(output_file, 'w', encoding='utf-8') as fout:
            with open(input_file, encoding='utf-8') as fin:
                for line in fin:
                    sample = json.loads(line.strip())
                    for field in fields:
                        sample[field] = fun(sample[field])
                    fout.write(json.dumps(sample) + '\n')

    def segment_data(self):
        self.prepro_data(self.raw_file, self.segment_file, self.prepro_fields, self.tokenizer.segment)

    def ngram_data(self):
        get_ngram = partial(self.ngram_transform.get_ngrams, n=self.ngram_num)
        self.prepro_data(self.segment_file, self.ngram_file, self.prepro_fields, get_ngram)

    def load_data(self, cat='ngram'):
        file = self.ngram_file
        if cat == 'segment':
            file = self.segment_file
        print(file)
        with open(file, encoding='utf-8') as fin:
            for line in fin:
                sample = json.loads(line.strip())
                self.data_set.append(sample)

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
