import time
import requests
import random
import os
from lxml import etree


base_url = "http://www.dianping.com"
base_path = "/root/lsy/graduateDesign/"
user_agents = [
		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 OPR/26.0.1656.60',
		'Opera/8.0 (Windows NT 5.1; U; en)',
		'Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1) Gecko/20061208 Firefox/2.0.0 Opera 9.50',
		'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; en) Opera 9.50',
		'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0',
		'Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10',
		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2 ',
		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36',
		'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
		'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16',
		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36',
		'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/2.0 Safari/536.11',
		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER',
		'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)',
		'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 SE 2.X MetaSr 1.0',
		'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; SE 2.X MetaSr 1.0) ',
	]

proxy_host = 'dyn.horocn.com'
proxy_port = 50000
proxy_username = 'XDSE1626626118039305'
proxy_pwd = 'Z0OflY7JiniX'
proxies = {
    'http': 'http://{}:{}@{}:{}'.format(proxy_username, proxy_pwd, proxy_host, proxy_port),
}
'''
proxy_list = []
f = open('ipList', 'r')
line = f.readline()
rank = 0

while line:
    line = line.strip().strip('\n')
    #ip = line.split('\t')[0]
    #port = line.split('\t')[1]
    #proxy = 'http://%s:%s' % (ip, port)
    proxy = 'http://%s' % (line)
    proxy_list.append(proxy)
    line = f.readline()
'''

def check_usable(proxy):
    try:
        url = 'http://baidu.com/s?wp=ip'
        proxies = {
            'http': proxy
        }
        try:
            r = requests.get(url, proxies=proxies, timeout=5)
            return proxy
        except:
            return None
    except Exception as e:
        print(e)


def get(url):
    sec = random.randint(1,10) 
    time.sleep(sec)
    '''
    while True:
        proxy = random.choice(proxy_list)
	#print proxy
        ret = check_usable(proxy)
	#print ret
        if ret is not None:
            break
    proxies = {
        'http': proxy
    }
    '''
    headers = {'User-Agent': random.choice(user_agents)}
    r = requests.get(url, headers=headers, proxies=proxies)
    if r.status_code == 200:
	with open('source.txt', 'a') as f:
	     f.write(url + '\t' + r.content + '\n')
        return r.content
    else:
        print("can't get url response")


def get_img(url, d_name, name):
    global rank
    path = '%s%s/%s' %(base_path, name, d_name)
    if not os.path.exists(path):
        os.makedirs(path)
    img_path = '%s/%d' % (path, rank)
    res = get(url)
    with open(img_path, 'wb') as f1:
        f1.write(res)
    rank += 1


def parse_dish(url, d_name, name):
    url = base_url + url
    r = ''
    retryNum = 3
    while retryNum > 0:
        r = get(url)
        if r:
            print('get url -- ' + url)
            break
        retryNum -= 1
    if retryNum != 0:
        r = r.decode('utf-8')
	html_obj = etree.HTML(r)
        img_url_list = html_obj.xpath('//div[@class="img"]/a/img/@src')
        if img_url_list:
            for src in img_url_list:
                src = src.strip().strip('\n').strip('\r\n')
		if "https" in src:
		    src = src.replace("https", "http")
                get_img(src, d_name, name)
        else:
            print('parse error -- ' + url)
    else:
        print("can't get url -- " + url)

def parse_shop(url, name):
    url = url+'/photos'
    r = ''
    retryNum = 3
    while retryNum > 0:
        r = get(url)
        if r:
            print('get url -- ' + url)
            break
        retryNum -= 1
    if retryNum != 0:
	r = r.decode('utf-8')
	with open('sample.html', 'w') as f:
		f.write(r)
        html_obj = etree.HTML(r)
        dish_url_list = html_obj.xpath('//div[@id="photoNav"]/ul/li[3]/dl/dd/a/@href')
        dish_name_list = html_obj.xpath('//div[@id="photoNav"]/ul/li[3]/dl/dd/a/@title')
	print('%d %d' % (len(dish_url_list), len(dish_name_list)))
        if len(dish_url_list) == len(dish_name_list):
            for d_url, d_name in zip(dish_url_list, dish_name_list):
                d_url = d_url.strip().strip('\n').strip('\r\n')
		if 'https' in d_url:
		    d_url = d_url.replace("https", "http")
                d_name = d_name.strip().strip('\n').strip('\r\n')
                parse_dish(d_url, d_name, name)
        else:
            print('parse error -- ' + url)
    else:
        print("can't get url -- " + url)

def parse_branch(ori_url, name):
    for p in range(1, 2):
        url = "%sp%d" % (ori_url, p)
        r = ''
        retryNum = 3
        while retryNum > 0:
            r = get(url)
            if r:
                print('get url -- ' + url)
                break
            retryNum -= 1
        if retryNum != 0:
	    r = r.decode('utf-8')
            html_obj = etree.HTML(r)
            shop_url_list = html_obj.xpath('//div[@class="pic"]/a/@href')
            if shop_url_list:
                for s_url in shop_url_list:
                    s_url = s_url.strip().strip('\n').strip('\r\n')
                    if "https" in s_url:
			s_url = s_url.replace('https', 'http')
		    parse_shop(s_url, name)
            else:
                print('parse error -- ' + url)
        else:
            print("can't get url -- " + url)


def main():
    ori_url = "http://www.dianping.com/beijing/ch10/g1338"
    retryNum = 3
    r = ''
    while retryNum > 0:
        r = get(ori_url)
        if r:
            print('get url -- '+ori_url)
            break
        retryNum -= 1
    if retryNum != 0:
	r = r.decode('utf-8')
        html_obj = etree.HTML(r)
        branch_url_list = html_obj.xpath('//div[@id="classfy"]/a/@href')
        branch_name_list = html_obj.xpath('//div[@id="classfy"]/a/span/text()')
        if len(branch_url_list) == len(branch_name_list):
            for url, name in zip(branch_url_list, branch_name_list):
		url = url.strip().strip('\n').strip('\r\n')
		if "https" in url:
		    url = url.replace("https", "http")
                name = name.strip().strip('\n').strip('\r\n')
                parse_branch(url, name)
        else:
            print('parse error -- ' + ori_url)
    else:
        print("can't get url -- " + ori_url)

if __name__ == '__main__':
    main()





