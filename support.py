import jieba
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import jieba.analyse
import numpy as np
from pypinyin import lazy_pinyin
from pymongo import MongoClient
mix_file_name = 'mix_file'
sort_sports = {'田径': ['短跑', '撑杆跳高', '跳高', '三级跳', '跳远', '链球', '标枪', '铁饼', '铅球',
                        '马拉松', '竞走', '长跑', '障碍跑', '接力跑', '跨栏'],
               '球类': ['乒乓球', '羽毛球', '水球', '篮球', '足球', '手球', '棒球', '垒球', '排球',
                        '曲棍球', '网球', '沙滩排球'],
               '游泳类': ['游泳'],
               '跳水类': ['跳水'],
               '体操类': ['竞技体操', '蹦床'],
               '自行车类': ['山地自行车'],
               '对抗类': ['柔道', '跆拳道', '击剑', '拳击', '摔跤'],
               '举重类': ['举重'],
               '射击类': ['射箭', '射击'],
               '船艇类': ['赛艇', '帆船', '皮划艇'],
               '马术类': ['马术']}  # 11个大类，44个小类
categories = {'sports': sort_sports}  # categories sort item keyword


def insert_doc(docs):
    with MongoClient('localhost', 27017) as client:
        db = client.local
        col = db.key_weight
        col.insert(docs)


def get_stop_words(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]


def get_text(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return str(file.read()).strip().replace(' ', '').replace('\n', '').replace('\r', '')


def text_rank(text):
    """
    这个停用词表不好，最终效果不太好
    :param text: string
    :return:
    """
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, window=2)
    print('关键词：')
    for item in tr4w.get_keywords(20, word_min_len=1):
        print(item.word, item.weight)
    print('关键短语：')
    for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num=2):
        print(phrase)
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    print()
    print('摘要：')
    for item in tr4s.get_key_sentences(num=3):
        print(item.index, item.weight, item.sentence)


def tf_idf(corpus, category):
    # keyword = get_stop_words('sports_items.txt')
    assert len(keyword) == len(corpus)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=1)
    vectorizer.fit_transform(corpus)
    feature_name = list(vectorizer.get_feature_names())
    feature_weight = vectorizer.fit_transform(corpus).toarray()
    a = list(np.argsort(feature_weight))
    out = []
    for x in range(len(a)):
        indices = list(reversed(a[x][-25:]))
        # print(keyword[x])
        for y in indices:
            # print(feature_name[y], feature_weight[x][y])
            for i in categories[category]:
                if keyword[x] in categories[category][i]:
                    out.append({'category': category,
                                'sort': i,
                                'item': keyword[x],
                                'keyword': feature_name[y],
                                'weight': feature_weight[x][y]})
                    break
    return out


def spider_sports():
    """
    :return: [每个url下的原始文本]
    """
    import requests
    import re
    user_agent = {'User-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleW'
                                'ebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36'}
    # keyword = get_stop_words('sports_items.txt')
    url_list = list(map(lambda i: 'https://baike.baidu.com/item/'+str(i), keyword))
    print(url_list)
    text = []
    for url in url_list:
        # print(url)
        a = requests.get(url, headers=user_agent)
        a.encoding = 'utf-8'
        b = re.findall('锁定<(.*?)>词条统计', a.text, re.S)
        assert len(b) == 1
        text.append(b[0])
        # break
    return text


def extract_ch(text):
    """
    :param text: 原始文本，可以是list也可以是str
    :return: 将原始文本提取出中文，list()
    """
    import re
    p2 = re.compile(u'[^\u4e00-\u9fa5]')  # represent chinese encode
    if isinstance(text, str):
        zh = ','.join(' '.join(p2.split(text)).strip().split())
        return zh
    elif isinstance(text, list):
        zh = list(map(lambda a: ','.join(' '.join(p2.split(a)).strip().split()), text))
        return zh


def w_file(keywords, text):
    """

    :param keywords: 名称列表
    :param text: 文本列表
    :return: 写入文件
    """
    from pypinyin import lazy_pinyin
    assert len(keywords) == len(text)
    for i in range(len(text)):
        with open('sports/'+''.join(lazy_pinyin(keywords[i]))+'.txt', 'wb') as file:
            a = text[i]
            file.write(a.encode('utf-8'))


def seg_stop_mix(corpus_file_list):
    out = []
    for i in corpus_file_list:
        # text_1 = get_text(i)
        # seg_list_1 = list(jieba.cut(get_text(i), cut_all=False))  # 精确模式
        out.append(' '.join(list(filter(lambda a: a not in stop_words and len(a) >= 2,
                                        list(jieba.cut(get_text(i), cut_all=False))))))
    return out


if __name__ == '__main__':
    stop_words = get_stop_words('stop.txt')
    keyword = get_stop_words('sports_items.txt')
    # w_file(get_stop_words('sports_items.txt'), extract_ch(spider_sports()))  # 爬取后提取中文，并生成文件写入
    documents = tf_idf(seg_stop_mix(['sports/'+''.join(lazy_pinyin(keyword[i]))+'.txt' for i in range(len(keyword))]),
                       'sports')  # 得到category下的tf-idf值，返回字典{}插入数据库
    insert_doc(documents)
