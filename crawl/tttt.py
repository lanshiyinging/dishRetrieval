import sys
import time
import hashlib
import requests
# import grequests
from lxml import etree

_version = sys.version_info

is_python3 = (_version[0] == 3)

orderno = "ZF2019315858bCzeHR"
secret = "101436c38a554304a9047d92638a0cc5"

ip = "forward.xdaili.cn"
port = "80"

ip_port = ip + ":" + port

timestamp = str(int(time.time()))  
string = ""
string = "orderno=" + orderno + "," + "secret=" + secret + "," + "timestamp=" + timestamp

if is_python3:                          
    string = string.encode()

md5_string = hashlib.md5(string).hexdigest()            
sign = md5_string.upper()                    
print(sign)
auth = "sign=" + sign + "&" + "orderno=" + orderno + "&" + "timestamp=" + timestamp

print(auth)
proxy = {"http": "http://" + ip_port, "https": "https://" + ip_port}
headers = {"Proxy-Authorization": auth}
for i in range(10):
	r = requests.get("https://httpbin.org/ip", headers=headers, proxies=proxy, verify=False,allow_redirects=False)
	print(r.status_code)
	print(r.content)
	print(r.status_code)
'''
if r.status_code == 302 or r.status_code == 301 :
    loc = r.headers['Location']
    url_f = "https://www.tianyancha.com" + loc
    print(loc)
    r = requests.get(url_f, headers=headers, proxies=proxy, verify=False, allow_redirects=False)
    print(r.status_code)
    prinit(r.text)
'''
