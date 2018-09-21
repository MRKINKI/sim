import json
from .tool.ngram.ngram_transform import NgramTransform
from .tfidf.build_tfidf import BuildTfidf


class Dataset:
    def __init__(self, conf):
        self.ngram_transform = NgramTransform()
        self.buid_tfidf = BuildTfidf()
        self.tfidf_metadata_file = conf.get('path', 'tfidf_metadata_file')
        self.data_set = self.load_data(conf.get('path', 'prepro_file'))
        self.ngram_num = conf.get('model', 'ngram_num')
        self.word_key = conf.get('key', 'word')
        self.id_key = conf.get('key', 'id')
        self.ngram_key = conf.get('key', 'ngram')

    def load_data(self, file):
        data = []
        with open(file, encoding='utf-8') as fin:
            for line in fin:
                sample = json.loads(line.strip())
                data.append(sample)
        return data

    def ngram_prepro(self, word_key, ngram_key):
        for sample in self.data_set:
            words = sample[word_key]
            sample[ngram_key] = self.ngram_transform.get_ngrams(words, n=self.ngram_num)

    def build_tfidf(self):
        self.ngram_prepro(self.word_key, self.ngram_key)
        self.buid_tfidf.build(self.data_set, self.tfidf_metadata_file)
