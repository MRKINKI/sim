from .corenlp import StanfordCoreNLP
import collections


class CoreNLPTokenizer:
    def __init__(self, host='http://192.168.1.145'):
        self.corenlp = StanfordCoreNLP(path_or_host=host)

    def tokenize(self, text):
        deps, words, postags, netags, _ = self.corenlp.dependency_parse(text)
        dependencys = collections.Counter([t[0] for t in deps])
        if dependencys['ROOT'] > 1:
            return [], [], [], []
        return deps, words, postags, netags

    def segment(self, text, character=False):
        if not text.strip():
            return []
        elif character:
            return list(text)
        else:
            return self.corenlp.word_tokenize(text)
