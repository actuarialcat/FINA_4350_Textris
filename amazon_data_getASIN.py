import requests

url = "https://amazon-product-reviews-keywords.p.rapidapi.com/product/search"

headers = {
    'x-rapidapi-key': "9437f1c831msh6aae5cf355190f8p1d5077jsn4b8e761db189",
    'x-rapidapi-host': "amazon-product-reviews-keywords.p.rapidapi.com"
    }

intel_queries = [
    {"keyword":"intel cpu","category":"aps","country":"US"},
    {"keyword":"intel i3","category":"aps","country":"US"},
    {"keyword":"intel i5","category":"aps","country":"US"},
    {"keyword":"intel i7","category":"aps","country":"US"},
    {"keyword":"intel i9","category":"aps","country":"US"}
]

amd_queries = [
    {"keyword":"amd cpu","category":"aps","country":"US"},
    {"keyword":"amd ryzen3","category":"aps","country":"US"},
    {"keyword":"amd ryzen5","category":"aps","country":"US"},
    {"keyword":"amd ryzen7","category":"aps","country":"US"},
    {"keyword":"amd ryzen9","category":"aps","country":"US"}
]

f = open("intel_product_asin.txt", "a")
asin_list = []
for query in intel_queries:
    response = requests.request("GET", url, headers=headers, params=query)
    data = response.json()
    for product in data['products']:
        asin = product['asin']
        if asin not in asin_list:
            f.write(asin + '\n')
            asin_list.append(asin)
f.close()

f = open("amd_product_asin.txt", "a")
asin_list = []
for query in amd_queries:
    response = requests.request("GET", url, headers=headers, params=query)
    data = response.json()
    for product in data['products']:
        asin = product['asin']
        if asin not in asin_list:
            f.write(asin + '\n')
            asin_list.append(asin)
f.close()