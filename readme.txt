舆情挖掘与分析
1、搜集信息，文本预处理，例如：https://so.eastmoney.com/News/s?keyword=%E6%AC%A7%E8%8F%B2%E5%85%89
1）用jieba库实现中文分词
import jieba

text = "我喜欢吃苹果"
seg_list = jieba.cut(text, cut_all=False)
print(" ".join(seg_list))

2）停用词过滤&正则表达式过滤
import re
import nltk

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
text = "This is an example text for stopwords."
tokens = re.findall('\w+', text)
filtered_tokens = [token for token in tokens if not token.lower() in stopwords]
print(filtered_tokens)

2、特征提取
1）情感分析：用TextBlob库实现情感分析，polarity的值表示情感极性，范围在-1到1之间，值越大表示正面情感越强。
from textblob import TextBlob

text = "This is a happy day."
blob = TextBlob(text)
polarity = blob.sentiment.polarity
print(polarity)

2）主题模型：用gensim库实现主题模型
from gensim import corpora, models

texts = [["this", "is", "an", "example", "text"], ["another", "example", "text"]]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
for topic in lda.print_topics(num_words=3):
    print(topic)

3）关键词抽取，用TextRank算法实现关键词抽取
import jieba.analyse

text = "我爱北京天安门，天安门上太阳升。"
keywords = jieba.analyse.textrank(text, withWeight=True, topK=3)
for keyword, weight in keywords:
    print(keyword, weight)

4）生成词云
from wordcloud import WordCloud
# 生成词云
def create_word_cloud(word_dict):
    # 支持中文, SimHei.ttf可从以下地址下载：https://github.com/cystanford/word_cloud
    wc = WordCloud(
        font_path="./source/SimHei.ttf",
        background_color='white',
        max_words=25,
        width=1800,
        height=1200,
    )
    word_cloud = wc.generate_from_frequencies(word_dict)
    # 写词云图片
    word_cloud.to_file("wordcloud2.jpg")
    # 显示词云文件
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.show()

# 根据词频生成词云
create_word_cloud(word_dict)


3、事件检测，根据网络上的信息和趋势，识别和预测可能发生的突发事件，为决策者提供及时有效的决策依据
用基于机器学习和深度学习的方法实现事件检测
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["This is an example text for event detection.", "Another example text."]
labels = [0, 1]
vectorizer = TfidfVectorizer()
X = vectorizer

4、实现相似度分析算法后，需要对模型进行测试以验证其效果
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine_similarity

# 生成模拟数据
text1 = "文本1：你好，我是一个人工智能助手。"
text2 = "文本2：你好，我也是一个人工智能助手。"

# 分词
text1_words = [word.lower() for word in text1.split()]
text2_words = [word.lower() for word in text2.split()]

# 去除停用词
stopwords = set(pd.read_csv('stopwords.txt', header=None)[0])
text1_words = [word for word in text1_words if word.lower() not in stopwords]
text2_words = [word for word in text2_words if word.lower() not in stopwords]

# 词干化
text1_words = [' '.join(word for word in text1_words)[:-1] for word in text1_words]
text2_words = [' '.join(word for word in text2_words)[:-1] for word in text2_words]

# 计算余弦相似度
sim_1 = cosine_similarity(text1_words, text2_words)[0][0]
sim_2 = cosine_similarity(text2_words, text1_words)[0][0]

print(f"余弦相似度: {sim_1}")
print(f"皮尔逊相关系数: {sim_2}")