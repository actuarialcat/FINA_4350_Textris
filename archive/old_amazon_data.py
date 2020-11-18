import requests
import json
import time

asin_url = "https://amazon-product-reviews-keywords.p.rapidapi.com/product/search"
review_url = "https://amazon-product-reviews-keywords.p.rapidapi.com/product/reviews"

headers = {
    'x-rapidapi-key': "e3b0bd0f9fmsh9227e61a9bcc18bp1660edjsn34a594df309a",
    'x-rapidapi-host': "amazon-product-reviews-keywords.p.rapidapi.com"
}

intel_asin_querystrings = [
    {"keyword":"intel cpu","category":"aps","country":"US"},
    {"keyword":"intel i3","category":"aps","country":"US"},
    {"keyword":"intel i5","category":"aps","country":"US"},
    {"keyword":"intel i7","category":"aps","country":"US"},
    {"keyword":"intel i9","category":"aps","country":"US"}
]

amd_asin_querystrings = [
    {"keyword":"amd cpu","category":"aps","country":"US"},
    {"keyword":"amd ryzen3","category":"aps","country":"US"},
    {"keyword":"amd ryzen5","category":"aps","country":"US"},
    {"keyword":"amd ryzen7","category":"aps","country":"US"},
    {"keyword":"amd ryzen9","category":"aps","country":"US"}
]

intel_asin_list = []
for querystring in intel_asin_querystrings:
    response = requests.request("GET", asin_url, headers=headers, params=querystring)
    data = response.json()
    for product in data['products']:
        asin = product['asin']
        if asin not in intel_asin_list:
            intel_asin_list.append(asin)
intel_asin_list.sort()

amd_asin_list = []
for querystring in amd_asin_querystrings:
    response = requests.request("GET", asin_url, headers=headers, params=querystring)
    data = response.json()
    for product in data['products']:
        asin = product['asin']
        if asin not in amd_asin_list:
            amd_asin_list.append(asin)
amd_asin_list.sort()

for asin in intel_asin_list:
    asin = asin.rstrip("\n")
    time.sleep(1)
    review_querystring = {"asin":asin,"variants":"1","country":"US"}
    response = requests.request("GET", review_url, headers=headers, params=review_querystring)
    data = response.json()
    with open("Product review for AMD " + asin, "a") as outfile:
        json.dump(data, outfile)

for asin in amd_asin_list:
    asin = asin.rstrip("\n")
    time.sleep(1)
    review_querystring = {"asin":asin,"variants":"1","country":"US"}
    response = requests.request("GET", review_url, headers=headers, params=review_querystring)
    data = response.json()
    with open("Product review for AMD " + asin, "a") as outfile:
        json.dump(data, outfile)
