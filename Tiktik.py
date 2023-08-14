# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import requests  # 导入requests库
from urllib import request
from bs4 import BeautifulSoup
import time
# 图片转化为pdf文件
import glob
import fitz
import os


# 文章网址，通过文件读取,下载图片后拼成一个pdf文件，C:\Users\user\Downloads\

def getUrls(file_name):
    try:
        path = os.getcwd() + '\\' + file_name
        print(path + '\n')
        file = open(path, encoding='utf-8')
        return file.readlines()
    finally:
        file.close()


def getPic(path, url_add):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
    document_url = url_add
    page = request.Request(document_url, headers=headers)
    page_info = request.urlopen(page).read().decode('utf-8')
    soup = BeautifulSoup(page_info, 'html.parser')
    texts = soup.find_all('img', class_="rich_pages wxw-img js_insertlocalimg")
    text2 = soup.find_all('img', class_="rich_pages wxw-img")
    if len(texts) < len(text2):
        texts = soup.find_all('img', class_="rich_pages wxw-img")

    title_soup = soup.find_all('h1', class_="rich_media_title")
    doc_title = str(title_soup[0].text).strip()  # 获取文章标题
    if doc_title.find("|") > 0:
        doc_name = doc_title[:doc_title.index("|")]
    else:
        doc_name = doc_title
    new_path = path + "\\" + doc_name
    data = []
    i = 0
    try:
        # 在C盘以只写的方式打开/创建一个名为 text 的txt文件
        file = open(base_path + '\\text.txt', 'w')
        doc = fitz.open()
        if os.path.exists(new_path):
            print("文件夹已存在！")
        else:
            os.mkdir(new_path)
        for title in texts:
            # 将内容写入txt中
            if title is None:
                print('title:is none')
            else:
                text = str(title)
                file.write(text + '\n')
                url_add = title.get('data-src')   # 获取图片链接地址
                data.append(url_add)
                i = i + 1
                pic_path = new_path + '\\' + str(i) + '.png'
                img = requests.get(url_add, headers=headers).content
                # url是img的url
                f = open(pic_path, 'wb')  # 打开一个二进制文件
                f.write(img)
                time.sleep(1)
                print(pic_path)
                if os.path.getsize(pic_path):
                    imgdoc = fitz.open(pic_path)  # 打开图片
                    pdfbytes = imgdoc.convert_to_pdf()  # 使用图片创建单页的 PDF
                    imgpdf = fitz.open("pdf", pdfbytes)
                    doc.insert_pdf(imgpdf)  # 将当前页插入文档
                    if os.path.exists(new_path + '\\' + doc_name + ".pdf"):
                        os.remove(new_path + '\\' + doc_name + ".pdf")
                    doc.save(new_path + '\\' + doc_name + ".pdf")  # 保存pdf文件
    finally:
        if file:
            # 关闭文件（很重要）
            file.close()
        if doc:
            doc.close()


if __name__ == '__main__':
    filename = "Titles.txt"
    base_path = os.getcwd()
    it = 0
    while it < len(getUrls(filename)):
        getPic(base_path, getUrls(filename)[it])
        it = it + 1
