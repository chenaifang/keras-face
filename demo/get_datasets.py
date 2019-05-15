#!/usr/bin/env python
# coding=utf-8
import requests
import os
import re
from bs4 import BeautifulSoup

import argparse
import hashlib
import base64
import gzip
import time
import io


class YirenSpider(object):
    headers = {
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36"}
    yiren_str = 'http://www.zybus.com/yiren/'
    #yiren_str = 'http: // www.zybus.com / dlyr / list_181_1.html'
    html_suffix = '.html'

    #返回网页的所有内容
    def get_html_data(self, url):
        text = requests.get(url, headers=self.headers)
        return text.content

    #将HTML文档转换为复杂的树形结构
    def get_soup(self, text):
        soup = BeautifulSoup(text, "lxml")
        #soup = BeautifulSoup(text, 'html.parser')
        return soup

    #得到艺人主页的集合
    def get_yiren_htmls(self, url):
        text = self.get_html_data(url)
        soup = self.get_soup(text)
        totals = soup.find_all('a')#得到网页上的所有<a herf:>,
        img_url_list = []
        for link in totals:
            a_link = link.get('href')#得到link这个a标签中的网址
            #startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
            if len(a_link) >= (len(self.yiren_str) + len(self.html_suffix)) and a_link.startswith(self.yiren_str):
                if len(img_url_list) == 0:
                    img_url_list.append(a_link)
                else:
                    if img_url_list.count(a_link) > 0:
                        pass
                    else:
                        img_url_list.append(a_link)
        return img_url_list

    #得到艺人图库
    def get_yirentuku_html(self,url):
        text = self.get_html_data(url)
        soup = self.get_soup(text)
        print(soup)
        totals = soup.find('a', class_='lh_30 show r')
        print(totals)
        a_link = totals.get('href')
        print(a_link)


    def get_timestr(self):
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        dt = time.strftime("%Y-%m-%d_%H:%M:%S", time_local)
        return dt

    def get_img_url(self, url):  # 得到 图片的 html 地址
        img_prefix = 'src'
        img_suffix = '.jpg'

        text = self.get_html_data(url)
        soup = self.get_soup(text)
        totals = soup.find_all('div', class_='yiren_big_pic')
        #print(totals)
        img_urls = []
        for link in totals:
            #print(link)
            img_src = str(link)
            #print(img_src)
            img_src_len = len(img_src)
            index = img_src.find(img_prefix)
            #print(index)
            index_r = img_src.find(img_suffix)
            #print(index_r)
            img_urls.append(img_src[index + 5:index_r + 4])
            #print(img_urls)
        # print(img_src[index+5:index_r+4])
        return img_urls

    def get_img_list(self,url):
        img_prefix = 'src'
        img_suffix = '.jpg'
        text = self.get_html_data(url)#url是艺人主页
        soup = self.get_soup(text)#将主页内容转换为树形结构
        #print("soup")
        #print(type(soup))
        #print(soup)
        #totals = soup.find_all('div',id="imgList")#在主页转换为的树形结构终查找id为imgList的div
        #totals = soup.find_all('div', class_="w960")
        #totals=(soup.find("div",class_="w1000")).findAll("img")
        totals = soup.find_all("img")
        img_urls = []
        num=0
        for link in totals:
            img_src = str(link)
            img_src_len = len(img_src)
            index = img_src.find(img_prefix)
            index_r = img_src.find(img_suffix)
            img_urls.append(img_src[index + 5:index_r + 4])
            num=num+1
            if num is 7:
                break
            #print("img_urls")
            #print(img_urls)
        # print(img_src[index+5:index_r+4])
        return img_urls

    def download_images(self, urls, dirpath,name):  # 保存到指定的 文件夹
        if not os.path.exists(dirpath+"/"+name):
            os.makedirs(dirpath+"/"+name)
        file_num=0

        for u in urls:
            u_img_data = self.get_html_data(u)
            suffix = 'jpg'
            # suffix = suffix[len(suffix) - 1]
            #time.sleep(1)
            #filename = str(self.get_timestr())
            file_num=file_num+1
            file_name=name+"_000"+str(file_num)
            with open(dirpath + "/" + name + '/' + str(file_name) + '.' + suffix, 'wb') as f:
                f.write(u_img_data)

    def spider(self, url, dirpath):
        htmllist = self.get_yiren_htmls(url)  # 得到主页 艺人的html 地址http://www.zybus.com/yiren/wangfei.html的集合
        for html in htmllist:
            #urls = self.get_img_url(html)  # 在某艺人的 主页上获取他那张最大的pic 图像,['http://img.zybus.com/uploads/131220/1-131220101346243.jpg']
            #print(html)   #每个HTML类似http://www.zybus.com/yiren/yeyixi.html
            html_split=html.split("/")
            html_name=html_split[-1]
            html_name_split=html_name.split(".")
            name=html_name_split[0]#艺人名字---------------------
            print(name)
            #self.get_yirentuku_html(html)
            urls = self.get_img_url(html)
            self.download_images(urls, dirpath,name)  # 保存下来
            pass


if __name__ == '__main__':
    url = 'http://www.zybus.com/omyr/'
    #url = 'http://www.zybus.com/yiren/'
    dir = './data/picture'

    Yiren = YirenSpider()
    Yiren.spider(url, dir)