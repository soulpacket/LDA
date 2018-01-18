from gensim import corpora, models
from support import get_stop_words, extract_ch
import jieba
import re
from pymongo import MongoClient
from collections import defaultdict
import operator
stop_words = get_stop_words('stop.txt')


def regex(string):
    """
    remove emoij: [ ]中的
    :param string:
    :return:
    """
    return ''.join(re.split('\[.*?\]', string, re.S))


def input_vector():
    """
    预处理后得到输入向量，[[words]]
    :return: LDA模型的输入向量
    """
    with open('lda_data/subingtian.txt', 'r', encoding='utf-8') as file:
        word_list = []
        for line in file:
            b = line.replace('🏓', '乒乓球').replace('🏅', '奖牌').replace(' ', '').strip()
            b = extract_ch(regex(b))
            seg_list = list(jieba.cut(b, cut_all=False))
            # print(seg_list)
            seg = list(filter(lambda a: a != '️' and a != '\u200b' and a != '\xa0' and len(a) >= 2, seg_list))
            word_list.append(list((filter(lambda a: a not in stop_words, seg))))
        return word_list


def lda_run(word_list):
    """
    run LDA
    :param word_list: [[],[]]
    :return: {topic: weight}
    """
    # print(word_list)
    word_dict = corpora.Dictionary(word_list)
    # print(dict(word_dict))
    corpus_list = [word_dict.doc2bow(text) for text in word_list]
    # print(corpus_list)
    # out = []
    lda = models.ldamodel.LdaModel(corpus=corpus_list, id2word=word_dict, num_topics=1, alpha='auto', passes=50)
    for pattern in lda.show_topics(formatted=False, num_words=10):
        # out.append(dict(pattern[1]))
        return dict(pattern[1])


def get_sort(topic):
    """
    计算权值相乘后排序，分类
    :param topic: lda计算得到的字典，{topic: weight}
    :return:item权值从高到低
    """
    with MongoClient('localhost', 27017) as client:
        col = client.local.key_weight
        item_dict = defaultdict(int)
        for i in col.find({}):
            if i['keyword'] in topic:
                item_dict[i['item']] += topic[i['keyword']]*i['weight']
        return sorted(item_dict.items(), key=operator.itemgetter(1), reverse=True)
if __name__ == '__main__':
    topics = lda_run(input_vector())
    print(topics)
    print(get_sort(topics))
