from .tool.ngram.ngram_transform import NgramTransform
from .tool import tokenize
from .tfidf.ranker import TfidfRanker
from .topic.ranker import TopicRanker
import os


class SemanticMatchingInference:
    def __init__(self, conf, method='tfidf'):
        self.ngram_transform = NgramTransform()
        self.tokenizer = tokenize.get_class('corenlp')()
        self.ngram_num = conf.getint('model', 'ngram_num')
        self.tfidf_metadata_path = conf.get('path', 'tfidf_metadata_path')
        self.tfidf_ranker = None

        if method == 'tfidf':
            tfidf_metadata_file = conf.get('path', 'tfidf_metadata_path')
            # self.tfidf_ranker = TfidfRanker(model_path=tfidf_metadata_file)
        elif method == 'topic':
            lda_metadata_file = conf.get('path', 'lda_metadata_file')
            self.topic_ranker = TopicRanker(model_path=lda_metadata_file)

    def tfidf_field(self, field):
        model_file = os.path.join(self.tfidf_metadata_path, field)
        self.tfidf_ranker = TfidfRanker(model_path=model_file)

    def segment(self, text):
        return self.tokenizer.segment(text)

    def get_ngrams(self, words):
        return self.ngram_transform.get_ngrams(words, n=self.ngram_num)

    def tfidf_rank_from_text(self, text, field, topk=5):
        words = self.segment(text)
        ngrams = self.get_ngrams(words)
        cands = self.tfidf_ranker.rank_from_text(ngrams, field, topk)
        return cands

    def tfidf_rank_from_id(self, text_id, topk=5):
        cands = self.tfidf_ranker.rank_from_id(text_id, topk)
        return cands

    def topic_rank_from_text(self, text, field, topk=5):
        words = self.segment(text)
        cands = self.topic_ranker.rank_from_text(words, field, topk)
        return cands
