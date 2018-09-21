from .corenlp_tokenizer import CoreNLPTokenizer


def get_class(name):
    if name == 'corenlp':
        return CoreNLPTokenizer
