import json
from urllib.request import urlopen, Request
import requests 
import pandas as pd
from tqdm import tqdm
from GetItemDetail import getItemDetail
from GetItemShopID import getItemShopID
import time
from pymongo import MongoClient
import argparse

parser = argparse.ArgumentParser(description='Crawl coment')

parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=1)
parser.add_argument('--nb_product', type=int, default=50)


args = parser.parse_args()


class Comment:
    def __init__(self,limit_commit=10) -> None:
        self.categoryList = pd.read_csv('Data/categoryList.csv')
        self.limit_commit = limit_commit
        self.connet_db()

    
    def connet_db(self):
        client = MongoClient("mongodb+srv://vukhanh09:Webmining@cluster0.arhqqik.mongodb.net/test")
        database = client['ShopeeComment']
        self.data_collections = database['comment']
    


    def comment_loader(self,start,end,nb_product):
        for item in range(start,end):
            topic = self.categoryList.catid.iloc[item]
            print(f'topic {item} "{self.categoryList.display_name.iloc[item]}" is runing...')
            for newest in range(0,nb_product,60):
                time.sleep(0.3)
                try:
                    ItemListID = getItemShopID(topic,newest)
                    data = self.get_comment(ItemListID.values,topic)
                    self.data_collections.insert_many(data)
                except Exception as e:
                    print(e)

    def get_comment(self,ItemListID,topic):
        features = ['orderid', 'itemid', 'cmtid', 'ctime', 'rating', 'userid', 'shopid', 
                'comment', 'rating_star', 'status', 'mtime', 'editable', 'opt', 'filter']

        url = 'https://shopee.vn/api/v2/item/get_ratings?filter=1&itemid={}&limit={}&offset={}&shopid={}&type={}'
        
        out_put_data = []
        
        def getFeature(data):
            comment = {}
            comment['orderid'] = data['orderid']
            comment['itemid'] = data['itemid']
            comment['cmtid'] = data['cmtid']
            comment['rating'] = data['rating']
            comment['userid'] = data['userid']
            comment['shopid'] = data['shopid']
            comment['comment'] = data['comment']
            comment['rating_star'] = data['rating_star']
            comment['status'] = data['status']
            comment['orderid'] = data['orderid']
            comment['mtime'] = data['mtime']
            comment['editable'] = data['editable']
            comment['opt'] = data['opt']
            comment['filter'] = data['filter']
            comment['topic'] = str(topic)
            return comment



        for type in range(1,6):
            print(f'Crawl rating: {type}')
            for itemid, shopid in tqdm(ItemListID):
                for offset in range(0, self.limit_commit, self.limit_commit):
                    try:
                        rep = requests.get(url.format(itemid,self.limit_commit ,offset, shopid,type)).content
                        data = json.loads(rep)['data']['ratings']
                        data = list(map(getFeature,data))
                        
                        out_put_data += data
                    except:
                        print("exception:", itemid, offset, shopid)    
        return out_put_data

if __name__ == "__main__":
    comment = Comment()
    comment.comment_loader(args.start,args.end,args.nb_product)