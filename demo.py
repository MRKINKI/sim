# -*- coding: utf-8 -*-

from main.tool import tokenize
from main.prepro.tokenize_data import TokenizeData
from main.dataset import Dataset
from main.tool.ngram.ngram_transform import NgramTransform
import configparser


if __name__ == '__main__':

    text = '人们通常只关注味道甜美的枇杷，很少会注意到其貌不扬的枇杷叶'
    conf = configparser.ConfigParser()
    conf.read("./conf.ini")

    ds = Dataset(conf)
    td = TokenizeData()
    nt = NgramTransform()
    tokenizer = tokenize.get_class('corenlp')()
    # td.run(raw_file, prepro_file, tokenizer)
    ds.build_tfidf()
    # print(da.data)

    words = tokenizer.segment(text)
    ngram = nt.get_ngrams(words, n=2)
    print(words)
    print(ngram)
