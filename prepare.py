#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import urllib
import re
import json
import pandas as pd
from urllib import request
from bs4 import BeautifulSoup
import os

URL = "Titles.txt"
RAW_FILE = "text.txt"


def load_data(file_name):
    file = None
    try:
        path = os.path.join(os.getcwd(), file_name)
        print(path + '\n')
        file = open(path, encoding='utf-8')
        return file.readlines()
    finally:
        file.close()


def get_stock_news(file_url):
    _url = file_url
    url_encoded = urllib.parse.quote(_url, safe=':/&=?')
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
    page = request.Request(url_encoded, headers=headers)
    page_info = request.urlopen(page).read().decode('utf-8')
    soup = BeautifulSoup(page_info, 'html.parser')
    msg = soup.text
    return msg


def get_json(msg):
    values = ""
    pattern = r"\{.*\}"
    dict_ = None
    text = msg
    result = re.findall(pattern, text)
    i = 0
    result3 = ""
    while i < len(result):
        if len(result[i]) > 0:
            print("匹配成功！msg：" + result[i])
            result2 = re.findall(pattern, result[i])
            j = 0
            while j < len(result2):
                result3 = result2[j]
                dict_ = json.loads(result3)
                values = dict_["result"]
                j += 1
        else:
            print("匹配失败！")
        i += 1
# 解析搜索结果页面的字典值cmsArticleWebOld，获取news列表
    news_info = ""
    df = None
    news_list = None
    if len(values["cmsArticleWebOld"]) > 0:
        k = 0
        news_list = values["cmsArticleWebOld"]
        while k < len(news_list):
            news_info += "{" + news_list[k]["date"] + ": " + news_list[k]["title"] + "},"
            print(news_info)
            k += 1
    if news_list is not None:
        df = pd.Series(news_list)
    else:
        df = 0
    return result3, df


def save_raw_data(text):
    # 在C盘以只写的方式打开/创建一个名为 text 的txt文件
    path = os.path.join(os.getcwd(), RAW_FILE)
    print(path + '\n')
    file = open(path, 'w', encoding='utf-8')
    file.write(text + '\n')
    file.close()


if __name__ == '__main__':
    filename = "Titles.txt"
    code_name = '华域汽车'
    file_url = 'https://so.eastmoney.com/news/s?keyword=' + code_name
    result = ""
    news = ""
    it = 0
    while it < len(load_data(filename)):
        msg = get_stock_news(load_data(filename)[it])
        result, news = get_json(msg)
        save_raw_data(result)
        it = it + 1

