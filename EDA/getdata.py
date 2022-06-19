from pymongo import MongoClient

client = MongoClient("mongodb+srv://vukhanh09:Webmining@cluster0.arhqqik.mongodb.net/test")
database = client['ShopeeComment']
data_collections = database['comment']
df =[]
for item in data_collections.find():
    df.append(item)
import pandas as pd
df = pd.DataFrame(df)
df[['comment','rating_star']].to_csv('data.csv',index=False)