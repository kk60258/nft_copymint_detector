import requests
from bs4 import BeautifulSoup
headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "max-age=0",
    "Host": "httpbin.org",
    "Sec-Ch-Ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"102\", \"Google Chrome\";v=\"102\"",
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": "\"Windows\"",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
    "X-Amzn-Trace-Id": "Root=1-62a9825e-06c217837ffaaab97d6ae030"
}

web = requests.get('https://opensea.io/assets', headers=headers)
soup = BeautifulSoup(web.text, "html.parser")
print(soup.prettify())  #輸出排版後的HTML內容

# imgs = soup.find_all('img')
# name = 0    #  設定圖片編號
# for i in imgs:
#     print(i['src'])
#     jpg = requests.get(i['src'])     # 使用 requests 讀取圖片網址，取得圖片編碼
#     # f = open(f'/content/drive/MyDrive/Colab Notebooks/download/test_{name}.jpg', 'wb')    # 使用 open 設定以二進位格式寫入圖片檔案
#     f = open(f'temp/test_{name}.jpg', 'wb')    # 使用 open
#     f.write(jpg.content)   # 寫入圖片的 content
#     f.close()              # 寫入完成後關閉圖片檔案
#     name = name + 1        # 編號增加 1