import requests
import random
import os
import sys
import time
import hashlib
from lxml import etree


headers = {
	'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0 Safari/605.1.15'
}

base_url = "https://www.douguo.com"
base_path = "/root/lsy/dishRetrieval/crawl/data/"
rank = 3665

def check_usable(proxy):
    try:
        url = 'http://baidu.com/s?wp=ip'
        proxies = {
            'http': proxy
        }
        try:
            r = requests.get(url, proxies=proxies)
            return proxy
        except:
            return None
    except Exception as e:
        print(e)


def get(url):
    time.sleep(random.randint(30, 60))
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            print("get url --- " + url)
            return r.content
        else:
            print("get url wrong --- " + url + str(r.status_code))
    except:
        print("can't get url --- " + url)


def get_img(url, name):
    time.sleep(2)
    global rank
    path = '%s%s' % (base_path, name)
    if not os.path.exists(path):
        os.makedirs(path)
    img_path = '%s/%d.jpg' % (path, rank)
    res = get(url)
    if res:
        with open(img_path, 'wb') as f1:
            f1.write(res)
        rank += 1


def main():
    dish_list = []
    f = open('addlist.txt', 'r')
    line = f.readline()
    while line:
        line = line.strip().strip('\n')
        dish_list.append(line)
        line = f.readline()
    for name in dish_list:
        url = "%s/search/recipe/%s" % (base_url, name)
        retryNum = 3
        r = ''
        while retryNum > 0:
            r = get(url)
            if r:
                break
            retryNum -= 1
        if retryNum != 0:
            r = r.decode('utf-8')
            html_obj = etree.HTML(r)
            img_url_list = html_obj.xpath('//a[@class="cook-img"]/@style')
            if img_url_list:
                for style in img_url_list:
                    style = style.strip().strip('\n').strip('\r\n').replace(')', '(')
                    url = style.split('(')[1]
                    url = url.strip().strip('\n').strip('\r\n')
                    get_img(url, name)
            else:
                print('parse error -- ' + url)
        time.sleep(1800)



if __name__ == '__main__':
    main()







