import time
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

import requests
from selenium.webdriver.common.keys import Keys
import random
import urllib.request
import os
import sys


def clean_filename(s):
    if not s:
        return ''
    bad_chars = '\\/:*?\"<>|,'
    for c in bad_chars:
        s = s.replace(c, '_')
    return s


if __name__ == '__main__':
    try:
        DIR = 'temp_0704'
        url = 'https://opensea.io/assets'

        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument('User-Agent=Mozilla/5.0')

        lists = []

        driver = webdriver.Chrome(options=chrome_options)
        # driver = webdriver.Firefox(executable_path="/path/to/geckodrive.exe")
        driver.get(url)
        print(driver.page_source.encode("utf-8"))



        # soup = BeautifulSoup(driver.text, "html.parser")
        # soup = BeautifulSoup(driver.page_source, "lxml")
        # print(soup.prettify())

        # WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CLASS_NAME, 'ltEKP'))


        # all_items = driver.find_elements(By.XPATH, '//a[@style="padding-bottom: 1px;"]')
        records = {}
        existed_file = os.listdir(DIR)
        existed_file = [f.split('.')[0] for f in existed_file]
        existed_dictionary = dict.fromkeys(existed_file, "local")
        records.update(existed_dictionary)
        print(f'BEGIN {len(records)}')
        num = 0
        error_dict = {}

        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        class_gridcell = 'sc-1xf18x6-0.haVRLx.Asset--loaded'#'sc-1xf18x6-0.sc-bnjqwy-0.haVRLx.floEcI.Asset--loaded'
        #sc-7qr9y8-0 sc-nedjig-1 iUvoJs fyXutN
        class_group = 'sc-7qr9y8-0.sc-nedjig-1.iUvoJs.fyXutN'#'sc-7qr9y8-0.iUvoJs'
        # class_anchor = 'sc-1pie21o-0.elyzfO.Asset--anchor'
        #
        class_footer = 'sc-7qr9y8-0.sc-nedjig-1.iUvoJs.eewaH'#'AssetCardFooter--name'

        #sc-1v86bfg-5 sc-1v86bfg-9 jjtpry ipJjmq AssetMedia--img
        class_image = "sc-1v86bfg-5.sc-1v86bfg-9.jjtpry.ipJjmq.AssetMedia--img"
        skip_count = 0
        nft_count = len(records)
        while True:
            all_items = driver.find_elements(By.CLASS_NAME, class_gridcell)
            # all_images = driver.find_elements(By.CLASS_NAME, class_image)
            # all_group = driver.find_elements(By.CLASS_NAME, class_group)
            # all_anchor = driver.find_elements(By.CLASS_NAME, class_anchor)
            # all_footer = driver.find_elements(By.CLASS_NAME, class_footer)
            #
            # check_exist = [links.get_attribute('alt') in records for links in all_items]
            # if all(check_exist) and len(all_items) > 0:
            #     print(f'EOF {len(records)}')
            #     break

            for items in all_items:
                # sleep_time = random.randint(10, 500)
                # time.sleep(float(sleep_time)/1000)
                group = items.find_element(By.CLASS_NAME, class_group)

                footer = items.find_element(By.CLASS_NAME, class_footer)
                image = items.find_element(By.CLASS_NAME, class_image).find_element(By.TAG_NAME, 'img')
                # print(f'process {group.text} {footer.text}')
                src = image.get_attribute('src')
                src_parse = src.split('/')
                src_file = src_parse[-1]
                extension = 'jpg' if '.' not in src_file else src_file.split('.')[-1]
                if '?' in extension:
                    extension = extension.split('?')[0]

                alt = group.text + '==' + footer.text
                if alt == '':
                    alt = src_file.split('.')[0]

                alt = clean_filename(alt)

                #TODO replace |
                if alt in records:
                    skip_count += 1
                    print(f'skip {skip_count} {alt}')
                    continue

                try:

                    urllib.request.urlretrieve(src, f'{DIR}/{alt}.{extension}')
                    nft_count += 1
                    print(f'{nft_count} {alt} {src}')
                    records.update({alt: src})
                    # jpg = requests.get(i['src'])
                    # f = open(f'temp/{alt}', 'wb')    # 使用 open
                    # f.write(src.content)   # 寫入圖片的 content
                    # f.close()              # 寫入完成後關閉圖片檔案
                except Exception as e:
                    print(f'{alt} {src} error')
                    error_dict.update({alt: src})
                    print(e)

            # if num > 3:
            #     break
            html = driver.find_element(By.TAG_NAME, 'html')
            html.send_keys(Keys.PAGE_DOWN)
            sleep_time = random.randint(1000, 3000)
            time.sleep(float(sleep_time)/1000)
            num += 1
        driver.quit()
        print(f'-' * 30)
        print(f'-' * 30)

    except Exception:
        print('Interrupted')


    for k, v in error_dict.items():
        print(f'{k} {v} error')