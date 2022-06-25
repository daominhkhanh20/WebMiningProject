import requests 
import json
import pandas as pd 
from tqdm import tqdm
url = "https://shopee.vn/api/v4/pages/get_homepage_category_list"
data = requests.get(url).content
data = json.loads(data)['data']["category_list"]

categoryList = pd.DataFrame(data)

categoryList.to_csv('./Data/categoryList.csv',index=False)