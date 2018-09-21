import json
from .tool.ngram.ngram_transform import NgramTransform
from .tfidf.build_tfidf import BuildTfidf


class Dataset:
    def __init__(self, args):
        self.ngram_transform = NgramTransform()
        self.buid_tfidf = BuildTfidf()
        # self.data_set = self.load_data(args.)
        self.ngram_num = 2

    def load_data(self, file):
        data = []
        with open(file, encoding='utf-8') as fin:
            for line in fin:
                sample = json.loads(line.strip())
                data.append(sample)
        return data

    def ngram_prepro(self, word_key='text_words', ngram_key='text_ngram'):
        for sample in self.data_set:
            words = sample[word_key]
            sample[ngram_key] = self.ngram_transform.get_ngrams(words, n=self.ngram_num)

    def build_tfidf(self):
        self.buid_tfidf.build(self.data_set, self.tfidf_metadata_file)
