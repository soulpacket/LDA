#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import re
import requests
import sys
import traceback
from datetime import datetime
from datetime import timedelta
from lxml import etree
from pypinyin import lazy_pinyin


class Weibo:
    cookie = {"Cookie": "_T_WM=0557942a372d93a9a669c7c5c7b095ab; TMPTOKEN=9bKl7rjLpUI1vJcIewwv4NsRCQc1"
                        "h5K2Kbfi2RMunGk6C4GAg59iqpG72mKYAULf; SCF=AkTVio6KdA0ss3ABWFHMcxqLJEvcC0bc"
                        "WDs5rSDwmsX5bmBi5dcEiOfyFXiwj19mWBvXfnwWfogqZiwZv_RIvcg.; SUB=_2A253TyleDeRhGeN"
                        "N7VsQ9irMwjSIHXVUs7cWrDV6PUJbkdANLXjZkW1NSY_dvyWlQOx_8jJ0gwYLQknj6MAu55QK; SUHB="
                        "04WuwNGTQ6OoYa; SSOLoginState=1514887438"}  # 将your cookie替换成自己的cookie

    # Weibo类初始化
    def __init__(self, user_id, filter=0):
        self.user_id = user_id  # 用户id，即需要我们输入的数字，如昵称为“Dear-迪丽热巴”的id为1669879400
        self.filter = filter  # 取值范围为0、1，程序默认值为0，代表要爬取用户的全部微博，1代表只爬取用户的原创微博
        self.username = ''  # 用户名，如“Dear-迪丽热巴”
        self.weibo_num = 0  # 用户全部微博数
        self.weibo_num2 = 0  # 爬取到的微博数
        self.following = 0  # 用户关注数
        self.followers = 0  # 用户粉丝数
        self.weibo_content = []  # 微博内容
        self.publish_time = []  # 微博发布时间
        self.up_num = []  # 微博对应的点赞数
        self.retweet_num = []  # 微博对应的转发数
        self.comment_num = []  # 微博对应的评论数

    # 获取用户昵称
    def get_username(self):
        try:
            url = "https://weibo.cn/%d/info" % self.user_id
            html = requests.get(url, cookies=self.cookie).content
            selector = etree.HTML(html)
            username = selector.xpath("//title/text()")[0]
            self.username = username[:-3]
            print(u"用户名: " + self.username)
        # except Exception, e:
        except:
            # print("Error: ", e)
            traceback.print_exc()

    # 获取用户微博数、关注数、粉丝数
    def get_user_info(self):
        try:
            url = "https://weibo.cn/u/%d?filter=%d&page=1" % (
                self.user_id, self.filter)
            html = requests.get(url, cookies=self.cookie).content
            selector = etree.HTML(html)
            pattern = r"\d+\.?\d*"

            # 微博数
            str_wb = selector.xpath(
                "//div[@class='tip2']/span[@class='tc']/text()")[0]
            guid = re.findall(pattern, str_wb, re.S | re.M)
            for value in guid:
                num_wb = int(value)
                break
            self.weibo_num = num_wb
            print(u"微博数: " + str(self.weibo_num))

            # 关注数
            str_gz = selector.xpath("//div[@class='tip2']/a/text()")[0]
            guid = re.findall(pattern, str_gz, re.M)
            self.following = int(guid[0])
            print(u"关注数: " + str(self.following))

            # 粉丝数
            str_fs = selector.xpath("//div[@class='tip2']/a/text()")[1]
            guid = re.findall(pattern, str_fs, re.M)
            self.followers = int(guid[0])
            print(u"粉丝数: " + str(self.followers))

        # except Exception, e:
        #     print "Error: ", e
        #     traceback.print_exc()
        except:
            traceback.print_exc()
    # 获取用户微博内容及对应的发布时间、点赞数、转发数、评论数

    def get_weibo_info(self):
        try:
            url = "https://weibo.cn/u/%d?filter=%d&page=1" % (
                self.user_id, self.filter)
            html = requests.get(url, cookies=self.cookie).content
            selector = etree.HTML(html)
            if selector.xpath("//input[@name='mp']"):
                page_num = 1
            else:
                page_num = int(selector.xpath(
                    "//input[@name='mp']")[0].attrib["value"])
            pattern = r"\d+\.?\d*"
            for page in range(1, page_num + 1):
                url2 = "https://weibo.cn/u/%d?filter=%d&page=%d" % (
                    self.user_id, self.filter, page)
                html2 = requests.get(url2, cookies=self.cookie).content
                selector2 = etree.HTML(html2)
                info = selector2.xpath("//div[@class='c']")
                if len(info) > 3:
                    for i in range(0, len(info) - 2):
                        # 微博内容
                        str_t = info[i].xpath("div/span[@class='ctt']")
                        weibo_content = str_t[0].xpath("string(.)").encode(
                            sys.stdout.encoding, "ignore").decode(
                            sys.stdout.encoding)
                        self.weibo_content.append(weibo_content)
                        print(u"微博内容：" + weibo_content)

                        # 微博发布时间
                        str_time = info[i].xpath("div/span[@class='ct']")
                        str_time = str_time[0].xpath("string(.)").encode(
                            sys.stdout.encoding, "ignore").decode(
                            sys.stdout.encoding)
                        publish_time = str_time.split(u'来自')[0]
                        if u"刚刚" in publish_time:
                            publish_time = datetime.now().strftime(
                                '%Y-%m-%d %H:%M')
                        elif u"分钟" in publish_time:
                            minute = publish_time[:publish_time.find(u"分钟")]
                            minute = timedelta(minutes=int(minute))
                            publish_time = (
                                datetime.now() - minute).strftime(
                                "%Y-%m-%d %H:%M")
                        elif u"今天" in publish_time:
                            today = datetime.now().strftime("%Y-%m-%d")
                            time = publish_time[3:]
                            publish_time = today + " " + time
                        elif u"月" in publish_time:
                            year = datetime.now().strftime("%Y")
                            month = publish_time[0:2]
                            day = publish_time[3:5]
                            time = publish_time[7:12]
                            publish_time = (
                                year + "-" + month + "-" + day + " " + time)
                        else:
                            publish_time = publish_time[:16]
                        self.publish_time.append(publish_time)
                        print(u"微博发布时间：" + publish_time)

                        # 点赞数
                        str_zan = info[i].xpath("div/a/text()")[-4]
                        guid = re.findall(pattern, str_zan, re.M)
                        up_num = int(guid[0])
                        self.up_num.append(up_num)
                        print(u"点赞数: " + str(up_num))

                        # 转发数
                        retweet = info[i].xpath("div/a/text()")[-3]
                        guid = re.findall(pattern, retweet, re.M)
                        retweet_num = int(guid[0])
                        self.retweet_num.append(retweet_num)
                        print(u"转发数: " + str(retweet_num))

                        # 评论数
                        comment = info[i].xpath("div/a/text()")[-2]
                        guid = re.findall(pattern, comment, re.M)
                        comment_num = int(guid[0])
                        self.comment_num.append(comment_num)
                        print(u"评论数: " + str(comment_num))

                        self.weibo_num2 += 1

            if not self.filter:
                print(u"共" + str(self.weibo_num2) + u"条微博")
            else:
                print(u"共" + str(self.weibo_num) + u"条微博，其中" + str(self.weibo_num2) + u"条为原创微博")
        except :
            # print "Error: ", e
            traceback.print_exc()

    # 将爬取的信息写入文件
    def write_txt(self):
        try:
            result = ''
            for i in range(1, self.weibo_num2 + 1):
                text = self.weibo_content[i - 1] + "\n"
                result = result + text

            file_dir = os.path.split(os.path.realpath(__file__))[
                0] + os.sep + "lda_data"
            if not os.path.isdir(file_dir):
                os.mkdir(file_dir)
            file_path = file_dir + os.sep + "%s" % ''.join(lazy_pinyin(self.username)) + ".txt"
            f = open(file_path, "wb")
            f.write(result.encode(sys.stdout.encoding))
            f.close()
            # print(u"微博写入文件完毕，保存路径:" + file_path)
        except :
            # print "Error: ", e
            traceback.print_exc()

    # 运行爬虫
    def start(self):
        try:
            self.get_username()
            self.get_user_info()
            self.get_weibo_info()
            self.write_txt()
            # print(u"信息抓取完毕")
            # print("===========================================================================")
        except :
            # print "Error: ", e
            pass


def main():
    try:
        # 使用实例,输入一个用户id，所有信息都会存储在wb实例中
        user_id = 2155645503  # 可以改成任意合法的用户id（爬虫的微博id除外）
        filter = 1  # 值为0表示爬取全部微博（原创微博+转发微博），值为1表示只爬取原创微博
        wb = Weibo(user_id, filter)  # 调用Weibo类，创建微博实例wb
        wb.start()  # 爬取微博信息
        print(u"用户名：" + wb.username)
        print(u"全部微博数：" + str(wb.weibo_num))
        print(u"关注数：" + str(wb.following))
        print(u"粉丝数：" + str(wb.followers))
        print(u"最新一条原创微博为：" + wb.weibo_content[0])
        print(u"最新一条原创微博发布时间：" + wb.publish_time[0])
        print(u"最新一条原创微博获得的点赞数：" + str(wb.up_num[0]))
        print(u"最新一条原创微博获得的转发数：" + str(wb.retweet_num[0]))
        print(u"最新一条原创微博获得的评论数：" + str(wb.comment_num[0]))
    except:
        # print "Error: ", e
        traceback.print_exc()


if __name__ == "__main__":
    main()
