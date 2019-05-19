from selenium import webdriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
client = webdriver.Chrome(chrome_options=chrome_options)
client.get("https://www.baidu.com")
print(client.page_source.encode('utf-8'))
client.quit()
