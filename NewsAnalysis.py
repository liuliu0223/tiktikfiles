#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

'''
geometric库安装步骤
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install torch-geometric
'''

import pandas as pd
import numpy as np
import re
import jieba
from sklearn.model_selection import train_test_split
from zhon.hanzi import punctuation
import nltk
from nltk.corpus import stopwords
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
#import torch
#import torch.nn.functional as F
#from torch_geometric.data import Data
#from torch_geometric.nn import GCNConv
#import seaborn as sns
import matplotlib.pyplot as plt

import prepare

nltk.download('stopwords')
URL = "Titles.txt"
RAW_FILE = "text.txt"


# 数据预处理
def clean_text(text):
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除英文字符
    text = re.sub(r'[a-zA-Z]+', '', text)
    # 去除中文符号
    text = re.sub(r'[{}]+'.format(punctuation), '', text)
    # 小写化
    text = text.lower()
    return text


def tokenize_text(text):
    # 分词
    tokens = jieba.lcut(text)
    return tokens


def remove_stopwords(tokens):
    # 去除停用词
    stop_words = set(stopwords.words('chinese'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


# 图构建
def build_graph(data, threshold):
    G = nx.Graph()
    # 添加节点
    for i in range(len(data)):
        G.add_node(i)
    # 计算相似度
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            sim = cosine_similarity(data[i], data[j])[0][0]
            if sim > threshold:
                G.add_edge(i, j, weight=sim)
    return G

'''
# 图神经网络训练
def train_gcn_model(features, labels, edges):
    # 将特征和标签转换为张量格式
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    # 构建数据对象
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index)
    # 划分训练集和测试集
    train_idx, test_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42)
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask[test_idx] = True
    # 创建GCN层
    num_features = features.shape[1]
    num_classes = len(np.unique(labels))
    conv1 = GCNConv(num_features, 64)
    conv2 = GCNConv(64, num_classes)
    # 训练模型
    optimizer = torch.optim.Adam(params=[conv1.parameters(), conv2.parameters()], lr=0.01)
    for epoch in range(200):
        optimizer.zero_grad()
        out = conv2(F.relu(conv1(data.x, data.edge_index)))
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
    # 预测测试集标签
    pred = out.argmax(dim=1)
    test_acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()
    return pred.numpy(), test_acc


# 可视化和解释
def visualize_sentiment(pred, data):
    # 将预测结果转换为DataFrame格式
    df = pd.DataFrame({'id': range(len(data)), 'text': data, 'sentiment': pred})
    # 统计每个情感类别的数量，并按照从高到低的顺序排列
    counts = df.groupby('sentiment')['id'].count().reset_index(name='count')
    counts = counts.sort_values(by='count', ascending=False)
    # 创建热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.pivot(index='id', columns='sentiment', values='sentiment'), cmap='YlGnBu')
    plt.title('Sentiment Analysis Results')
    plt.xlabel('Sentiment')
    plt.ylabel('Text ID')
    # 显示结果
    plt.show()
'''

'''
在这里，我们使用图构建函数 build_graph() 基于文本向量计算相似性，并将相似度大于阈值的文本连接成一个无向图。
然后我们为每个节点（即每个新闻）分配情感标签，并使用 GCN 模型训练预测模型。最后，我们使用 visualize_sentiment() 函数创建热力图来可视化舆情分析结果。

需要注意的是，这个例子仅用于演示如何使用 Python 实现舆情分析，实际上要更加复杂和严格。
在实际应用中，需要使用合适的数据集、预处理、特征提取和机器学习方法，并对模型进行调试和优化，在不断改进和实验的基础上提高其准确性和鲁棒性。
此外，还需要考虑数据安全和隐私保护等问题，以确保所分析的信息不会被滥用或泄露。
'''

if __name__ == '__main__':
    result = ""
    news = ""
    it = 0
    news_df = None
    while it < len(prepare.load_data(URL)):
        msg = prepare.get_stock_news(prepare.load_data(URL)[it])
        result, df = prepare.get_json(msg)
        prepare.save_raw_data(str(result))
        it = it + 1
        pd.Series._append(to_append=df, ignore_index=True)

    # 读取数据
    text = prepare.load_data(RAW_FILE)
    # 数据预处理
    msg = clean_text(text)
    tokens = tokenize_text(msg)
    filtered_tokens = remove_stopwords(tokens)
    # 图构建
    word_vec_model = None  # 这里使用自己的词向量模型或使用预训练模型

    text_vectors = []
    for tokens in filtered_tokens:
        vectors = []
        for token in tokens:
            try:
                vector = word_vec_model[token]
                vectors.append(vector)
            except KeyError:
                pass
        if len(vectors) == 0:
            text_vectors.append(np.zeros((300,), dtype=np.float32))
        else:
            mean_vector = np.mean(np.array(vectors), axis=0)
            text_vectors.append(mean_vector)
'''
    threshold = 0.5
    G = build_graph(text_vectors, threshold)

    features = np.array(text_vectors)
    labels = df['sentiment'].astype('category').cat.codes.values
    edges = np.array(G.edges())
    _, test_acc = train_gcn_model(features, labels, edges)
    pred, _ = train_gcn_model(features, labels, edges)
    visualize_sentiment(pred, df['text'])
'''