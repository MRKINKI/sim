# -*- coding: utf-8 -*-

from main.tool import tokenize
from main.dataset import Dataset
from main.tool.ngram.ngram_transform import NgramTransform
from main.inference import SemanticMatchingInference
from main.classification.labeller import Labbeller
from main.classification.ovr import OVR
from main.classification.chain import Chain
import configparser
import json
import pickle
from time import time
import scipy.sparse as sp
import numpy as np


if __name__ == '__main__':

    text = '花菇历来被国人作为延年益寿的补品，是香菇中的上品，含有丰富的营养价值，可帮助调节人体新陈代谢，助消化'
    text = '冬天到了，吃什么比较好'
    text = '天气冷了吃什么比较好'
    text = '天气冷了'
    text = '小孩'
    text = '美容'
    text = '老公'

    conf = configparser.ConfigParser()
    conf.read("./conf.ini")

    ds = Dataset(conf)

    # # prepro

    # ds.segment_data('./data/baidu_recipe.json',
    #                 './data/baidu_recipe_segment.json',
    #                 ["title", "illustration", "practice", "materials"],
    #                 True)
    # ds.ngram_data('./data/baidu_recipe_segment.json',
    #               './data/baidu_recipe_ngram.json',
    #               ["title", "illustration", "practice", "materials"])
    #
    # ds.segment_data('./data/recipe_label_data.json',
    #                 './data/recipe_segment.json',
    #                 ["title", "illustration", "practice", "materials"],
    #                 True)
    # ds.ngram_data('./data/recipe_segment.json',
    #               './data/recipe_ngram.json', ["title", "illustration", "practice", "materials"])
    #
    # ds.load_data(cat='ngram')
    # ds.train_test_split()
    # tgt_fields = ['cuisine', 'prepro_time', 'degree', 'technology', 'taste', 'cook_time', 'food_form', 'meal']
    # src_fields = ["title", "illustration", "practice", "materials"]
    # ds.build_vocab(src_fields, tgt_fields, mc=100)
    ds.load_vocab()
    # ds.transform('./data/train.json', './data/train.data', src_fields, tgt_fields)
    # ds.transform('./data/test.json', './data/test.data', src_fields, tgt_fields)
    # ds.transform('./data/baidu_recipe_ngram.json', './data/baidu_recipe.data', src_fields, tgt_fields, predict=True)
    # ds.transform('./data/recipe_ngram.json', './data/msj_recipe.data', src_fields, tgt_fields, predict=True)

    chain_tgt_fields = ['technology', 'taste', 'cuisine', 'degree', 'cook_time', 'prepro_time', 'food_form', 'meal']
    ovr = OVR()
    chain = Chain()
    lab = Labbeller()
    chain.train('./data/train.data', './data/test.data', chain_tgt_fields, './data/chain.data')
    # ovr.train('./data/train.data', './data/test.data', './data/ovr.data')
    # df = lab.label('./data/baidu_recipe.json', './data/baidu_recipe.data', './data/chain.data', ds.vocab)

    # # build tfidf

    # ds.load_data(cat='ngram')
    # ds.load_data(cat='segment')
    # ds.build_tfidf_model()

    # # build lda

    # ds.load_data(cat='ngram')
    # ds.load_data(cat='segment')
    # ds.build_lda_model()

    # # embedding

    # ds.build_fasttext_model()
    # ds.build_word2vec_model()

    # word = '鸡肉'
    # model = pickle.load(open('./data/fasttext.model', 'rb'))
    # model = pickle.load(open('./data/w2v.model', 'rb'))
    # computer_vec = model.wv['甜']
    # print(computer_vec)
    # print(model.most_similar(word))

    # lda_meta_data = pickle.load(open('./data/lda_metadata.data', 'rb'))
    # doc_topic_matrix = lda_meta_data['illustration']['doc_topic_matrix']
    # term_topic_matrix = lda_meta_data['illustration']['term_topic_matrix']

    # # lda_index = lda_meta_data['illustration']['lda_index']
    # dictionary = lda_meta_data['illustration']['dictionary']
    # tokenizer = tokenize.get_class('corenlp')()
    # words = tokenizer.segment(text)
    # query_bow = [dictionary.doc2bow(words)]
    # lda_model = lda_meta_data['illustration']['lda_model']
    #
    # print(lda_model.log_perplexity(query_bow))

    # corpus_tfidf = lda_meta_data['illustration']['tfidf'][query_bow]
    # print(list(corpus_tfidf))

    # topics = lda_model.print_topics(num_topics=20, num_words=10)
    # vector = lda_model[corpus_tfidf]
    # print(list(vector))

    # term_topics_matrix = lda_model.get_topics()
    # print(sp.csr_matrix(term_topics_matrix))

    # sims = lda_index[vector]
    # print(sorted(enumerate(sims.squeeze()), key=lambda f: -f[1]))

    # topic_terms = lda_meta_data['illustration']['lda_model'].get_topic_terms(15, topn=10)
    #
    # print(words)
    # print(query_bow)
    # print(list(vector))
    # print([dictionary.id2token[t[0]] for t in topic_terms])

    smi = SemanticMatchingInference(conf, 'tfidf')

    # ngram = nt.get_ngrams(words, n=2)

    # print(ngram)

    # cands = smi.tfidf_rank_from_text(text, topk=3)
    # print(cands)

    # ["title", "efficacy", "illustration", "practice", "materials"]
    field = 'materials'
    text_id = '5ba322e307e6f05148b64860'
    # smi.tfidf_field(field)
    #
    t0 = time()
    # ds.build_smi_data()
    # cands = smi.tfidf_rank_from_id(text_id, topk=3)
    # cands = smi.topic_rank_from_text(text, field, topk=5)
    print(time()-t0)
    # print(cands)
