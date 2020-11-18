import re
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup

headers = {
        'authority': 'www.amazon.com',
        'pragma': 'no-cache',
        'cache-control': 'no-cache',
        'dnt': '1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'none',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-dest': 'document',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    }

cookies = {
    'session-id':'',
    'session-id-time':'',
    'session-token':''
}
f = open("CPU_ASIN.txt", "r")
for line in f:
    name = line.strip().split(",")[0]
    asin = line.strip().split(",")[1]
    
    pageNumber = 1
    all_title = []
    all_content = []
    all_date = []
    while True:
        commentLink = "https://www.amazon.com/product-reviews/{0}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber={1}".format(asin, str(pageNumber))
        r=requests.get(commentLink, headers=headers, cookies=cookies)
        
        s = BeautifulSoup(r.text, 'lxml') # Use ‘ lxml ‘ to parse the webpage.

        titles = s.find_all('a', class_="a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold")
        contents = s.find_all('span', class_="a-size-base review-text review-text-content")
        dates = s.find_all('span', class_="a-size-base a-color-secondary review-date")
        if not titles:
            print("end at", commentLink)
            break
        for title, content, date in zip(titles, contents, dates):
            all_title.append(title.getText())
            all_content.append(content.getText())
            all_date.append(date.getText())
        pageNumber += 1
        time.sleep(0.1)

    df = pd.DataFrame(list(zip(all_title, all_content, all_date)), columns=["title", "content", "date"])
    df.to_csv("Product review for {0}.csv".format(name))
f.close
