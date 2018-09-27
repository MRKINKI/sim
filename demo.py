# -*- coding: utf-8 -*-

from main.tool import tokenize
from main.dataset import Dataset
from main.tool.ngram.ngram_transform import NgramTransform
from main.inference import SemanticMatchingInference
import configparser
import json
import pickle
from time import time


if __name__ == '__main__':

    text = '花菇历来被国人作为延年益寿的补品，是香菇中的上品，含有丰富的营养价值，可帮助调节人体新陈代谢，助消化'
    text = '冬天到了，吃什么比较好'
    text = '天气冷了吃什么比较好'
    text = '牙痛'
    conf = configparser.ConfigParser()
    conf.read("./conf.ini")

    af = json.loads(conf.get('key', 'all_field'))

    # ds = Dataset(conf)
    # ds.segment_data()
    # ds.ngram_data()
    # ds.load_data(cat='segment')
    # ds.build_tfidf_model()
    # ds.build_lda_model()
    # ds.build_fasttext_model()
    # ds.build_word2vec_model()

    # word = '鸡肉'
    # model = pickle.load(open('./data/fasttext.model', 'rb'))
    # model = pickle.load(open('./data/w2v.model', 'rb'))
    # computer_vec = model.wv['甜']
    # print(computer_vec)
    # print(model.most_similar(word))

    # lda_meta_data = pickle.load(open('./data/lda_metadata.data', 'rb'))
    # # lda_index = lda_meta_data['illustration']['lda_index']
    # dictionary = lda_meta_data['illustration']['dictionary']
    # tokenizer = tokenize.get_class('corenlp')()
    # words = tokenizer.segment(text)
    # query_bow = [dictionary.doc2bow(words)]
    # corpus_tfidf = lda_meta_data['illustration']['tfidf'][query_bow]
    # lda_model = lda_meta_data['illustration']['lda_model']
    # topics = lda_model.print_topics(num_topics=20, num_words=10)
    # vector = lda_model[corpus_tfidf]
    # print(list(vector))
    #
    # sims = lda_index[vector]
    # print(sorted(enumerate(sims.squeeze()), key=lambda f: -f[1]))

    # topic_terms = lda_meta_data['illustration']['lda_model'].get_topic_terms(15, topn=10)
    #
    # print(words)
    # print(query_bow)
    # print(list(vector))
    # print([dictionary.id2token[t[0]] for t in topic_terms])

    smi = SemanticMatchingInference(conf, 'topic')

    # ngram = nt.get_ngrams(words, n=2)

    # print(ngram)

    # cands = smi.tfidf_rank_from_text(text, topk=3)
    # print(cands)

    # ["title", "efficacy", "illustration", "practice", "materials"]
    field = 'illustration'
    # text_id = '5ba3224507e6f00d78c4dca7'
    #
    t0 = time()
    # cands = smi.tfidf_rank_from_id(text_id, field, topk=3)
    cands = smi.topic_rank_from_text(text, field, topk=5)
    print(time()-t0)
    print(cands)
