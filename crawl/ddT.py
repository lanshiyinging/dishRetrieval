import requests
import random
import os
import sys
import time
import hashlib
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
#headers = {"Proxy-Authorization": auth}
headers = {
	'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0 Safari/605.1.15',
    "Proxy-Authorization": auth
}
for i in range(10):
        r = requests.get("https://httpbin.org/ip", headers=headers, proxies=proxy, verify=False,allow_redirects=False)
        print(r.status_code)
        print(r.content)
        print(r.status_code)
