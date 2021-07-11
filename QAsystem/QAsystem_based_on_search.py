import json
import nltk
from nltk.corpus import stopwords
import codecs
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from bert_embedding import BertEmbedding
import heapq
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def read_corpus():
    qlist = []
    alist = []
    filename = 'train-v2.0.json'
    datas = json.load(open(filename,'r'))
    data = datas['data']
    for d in data:
        paragraph = d['paragraphs']
        for p in paragraph:
            qas = p['qas']
            for qa in qas:
                #print(qa)
                #处理is_impossible为True时answers空
                if(not qa['is_impossible']):
                    qlist.append(qa['question'])
                    alist.append(qa['answers'][0]['text'])
    assert len(qlist) == len(alist)  # 确保长度一样
    return qlist, alist
qlist,alist = read_corpus()
# print(qlist[:2], "\n", alist[:2])

# ['When did Beyonce start becoming popular?',
#  'What areas did Beyonce compete in when she was growing up?']
#  ['in the late 1990s', 'singing and dancing']

def lowerCase(ori_list):
    return [q.lower() for q in ori_list]

def tokenizer(ori_list):
    #分词时处理标点符号
    SYMBOLS = re.compile('[\s;\"\",.!?\\/\[\]\{\}\(\)-]+')
    new_list = []
    for q in ori_list:
        words = SYMBOLS.split(q.strip())
        new_list.append(' '.join(words))
    return new_list

# nltk中stopwords包含what等，但是在QA问题中，这算关键词，所以不看作关键词
# 第一次使用nltk中的stopword之前，先单独运行 nltk.download('stopwords') 来下载
def removeStopWord(ori_list):
    new_list = []
    restored = ['what','when','which','how','who','where']
    english_stop_words = list(set(stopwords.words('english')))#
    for w in restored:
        english_stop_words.remove(w)
    for q in ori_list:
        sentence = ' '.join([w for w in q.strip().split(' ') if w not in
                             english_stop_words])
        new_list.append(sentence)
    return new_list

# 根据thres筛选词表，小于thres的词去掉
def removeLowFrequence(ori_list,vocabulary,thres=10):
    new_list = []
    for q in ori_list:
        sentence = ' '.join([w for w in q.strip().split(' ') if vocabulary[w] >= thres])
        new_list.append(sentence)
    return new_list

# 将数字统一替换,默认替换为#number
def replaceDigits(ori_list,replace='#number'):
    DIGITS = re.compile('\d+')
    new_list = []
    for q in ori_list:
        q = DIGITS.sub(replace,q)
        new_list.append(q)
    return new_list

# 定义处理一个单词的总的函数: 使用参数来控制各项清洗功能
def handle_sentence(word_list, isLowCase=True, isStopWord=True, isReplaceDigits=True):
    if isLowCase:
        word_list = lowerCase(word_list)
    word_list = tokenizer(word_list)
    if isStopWord:
        word_list = removeStopWord(word_list)
    if isReplaceDigits:
        word_list = replaceDigits(word_list)
    return word_list

new_qlist = handle_sentence(qlist)
print("new_qlist", new_qlist[:2])

# 清洗之前
# ['When did Beyonce start becoming popular?',
#  'What areas did Beyonce compete in when she was growing up?']
# 清洗之后
# ['when beyonce start becoming popular', 'what areas beyonce compete when growing']

# 首先先来看一下总词数
word_total = list()
for q in new_qlist:
    # 这里的q就是指的每一句话
    for w in q.split(' '):
        # w指的是每一个单词
        word_total.append(w)

word_total_unique = set(word_total)
# 输出总单词数
word_total.remove('')
print("word_total: ", len(word_total))
print("word_total_unique", len(word_total_unique))

# word_total:  562229
# word_total_unique 37731

# step1: 先统计词频
# dict_word_count:key：单词，value：词的出现次数
# 不用Counter的话，开始需要进行初始化
dict_word_count = {l: 0 for l in word_total}
for value in word_total:
    dict_word_count[value] += 1

# step2: 再根据词频统计出现1,2,3...n次的单词的个数
# 需要先把set保存，以此来作为字典的key
word_count_set = sorted(list(set(dict_word_count.values())))
dict_appear_count = {s: 0 for s in word_count_set}
for w, v in dict_word_count.items():
    dict_appear_count[v] += 1

# step3: 绘制出现次数的图，x轴为出现的次数；y轴为出现次数的单词数量；
x_data = list(dict_appear_count.keys())
y_data = list(dict_appear_count.values())
fig = plt.figure()  # 设置画布
ax1 = fig.add_subplot(111)
# 看前50个
k = 50
plt.plot(x_data[:k], y_data[:k])
ax1.set_xlabel(u'Word Appear Nums')
ax1.set_ylabel(u'Word Counts')
plt.show()

def removeLowFrequence(ori_list,vocabulary,thres=3):
    #根据thres筛选词表，小于thres的词去掉
    new_list = []
    for q in ori_list:
        sentence = ' '.join([w for w in q.strip().split(' ') if vocabulary[w] >= thres])
        new_list.append(sentence)
    return new_list
new_qlist = removeLowFrequence(new_qlist, dict_word_count)

vectorizer = TfidfVectorizer()  # 定一个tf-idf的vectorizer
X_tfidf = vectorizer.fit_transform(new_qlist)   # 结果存放在X矩阵

def get_least_numbers_big_data(alist, k):
    max_heap = []
    length = len(alist)
    # 当k传入的不满足范围时，返回为空
    if not alist or k <= 0 or k > length:
        return
    k -= 1
    for e in alist:
        if len(max_heap) <= k:
            heapq.heappush(max_heap, e)
        else:
            heapq.heappushpop(max_heap, e)
    return max_heap


def get_top_results_tfidf_noindex(query):
    # TODO 需要编写
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 对于用户的输入 query 首先做一系列的预处理(上面提到的方法)，然后再转换成tf-idf向量（利用上面的vectorizer)
    2. 计算跟每个库里的问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    input_seq = query
    input_vec = vectorizer.transform([input_seq])
    result = list(cosine_similarity(input_vec, X_tfidf)[0])
    top_values = sorted(get_least_numbers_big_data(result, 5), reverse=True)

    top_idxs = []
    len_result = len(result)
    dict_visited = {}
    for value in top_values:
        for index in range(len_result):
            if value == result[index] and index not in dict_visited:
                top_idxs.append(index)
                dict_visited[index] = True

    top_idxs = top_idxs[:5]

    return [alist[i] for i in top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案

'''
创建字典有两种方式--
1. 一种是开始就初始化，然后直接添加：inverted_idx =  {value:[] for value in word_total_unique}；
2. 另一种是，开始只定义dict()，然后通过if判断是否在词典中，在的话+1，否则赋值；
'''
inverted_idx = {word:[] for word in word_total_unique}
# 定一个一个简单的倒排表，是一个map结构。 循环所有qlist一遍就可以
for index, sentence in enumerate(new_qlist):
    for word in sentence.strip().split():
        inverted_idx[word].append(index)
inverted_idx.pop('')
# print(inverted_idx)

# 读取语义相关的单词
def get_related_words(file):
    dict_related = {}
    for line in open(file, mode='r', encoding='utf-8'):
        item = line.split(",")
        word, si_list = item[0], [value for value in item[1].strip().split()]
        dict_related[word] = si_list
    return dict_related

related_words = get_related_words('related_words.txt')
# 直接放在文件夹的根目录下，不要修改此路径。

def get_handled_input_seq(query):
    result = []
    for word in query.split():
        word = handle_sentence(word.split())
        if word != None:
            result += word
    return result


# 检查输入的问题并返回处理过的问题tf-idf用，返回为字符串
def check_query(query):
    input_seq = get_handled_input_seq(query)
    return ' '.join(input_seq)


# 利用倒排表和同义词获取相关的预料库中问题的序号
def get_related_sentences(query):
    # 得到的是分词过后每句话的列表
    input_seq = get_handled_input_seq(query)
    # 定义相关词list
    si_list = []
    for word in input_seq:
        # 得到每句话的词
        if word in related_words:
            for value in related_words[word]:
                si_list.append(value)

    total_list = input_seq
    for word in si_list:
        total_list.append(word)
    sentence_list = []
    for word in total_list:
        # 如果word在倒排表里
        if word in inverted_idx:
            sentence_list.extend(inverted_idx[word])
    return list(set(sentence_list))


def getTopIndexByResult(result):
    top_idxs = []
    top_values = sorted(get_least_numbers_big_data(result, 5), reverse=True)
    len_result = len(result)
    dict_visited = {}
    for value in top_values:
        for index in range(len_result):
            if value == result[index] and index not in dict_visited:
                top_idxs.append(index)
                dict_visited[index] = True
    return top_idxs


def get_top_results_tfidf(query):
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate （需要使用related_words).
    2. 对于候选文档，计算跟输入问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    # 将query转成字符串
    query = check_query(query)
    if query == "":
        print("please input a effect question", "")
        return None
    # 得到返回的序号
    sentence_list = get_related_sentences(query)

    top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表

    # 将输入的query转化为tf-idf向量
    input_seq = query
    input_vec = vectorizer.transform([input_seq])

    is_use_s_l = len(sentence_list) > 0

    if is_use_s_l == True:
        X_tfidf_si = []
        for id in sentence_list:
            X_tfidf_si.append(X_tfidf[id].toarray()[0])
        X_tfidf_si = np.array(X_tfidf_si)
        # csr_matrix根据行列索引到稀疏矩阵里的值
        result = list(cosine_similarity(input_vec, csr_matrix(X_tfidf_si))[0])
    else:
        result = list(cosine_similarity(input_vec, X_tfidf)[0])

    top_idxs = getTopIndexByResult(result)

    if is_use_s_l == True:
        top_idxs = [sentence_list[idx] for idx in top_idxs[:5]]
    else:
        top_idxs = top_idxs[:5]

    return [alist[i] for i in top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案


def test_output(question_list, iscorrect=False):
    for question in question_list:
        if question.strip() == "":
            print("Your question is empty")
            continue
        print(get_top_results_tfidf(question))

question_list = ["Where are you from ? ",
                 "What's the weather like today ?"]
test_output(question_list)
