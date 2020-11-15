import requests

url = "https://amazon-product-reviews-keywords.p.rapidapi.com/product/reviews"

querystring = {"asin":"","variants":"1","country":"US"}
headers = {
    'x-rapidapi-key': "9863a98825mshcec7e959565e04fp128a78jsna44a3953f2f1",
    'x-rapidapi-host': "amazon-product-reviews-keywords.p.rapidapi.com"
    }

f = open("amd_product_asin.txt", "r")
intel_asins = f.readlines()

for asin in intel_asins:
    asin = asin.rstrip('\n')
    querystring['asin'] = asin
    response = requests.request("GET", url, headers=headers, params=querystring)
    data = response.json()
    output = open("Review for AMD product " + asin, "a")
    for review in data['reviews']:
        content = review['review']
        output.write(content + '\n')
    output.close()
f.close()
"""




print(response.text)
"""