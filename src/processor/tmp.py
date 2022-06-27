import pandas as pd
import os
import codecs

from sqlalchemy import column
source_path = 'assets/_UIT-VSFC'
target_path = 'assets/data'
list_folder = ['dev','train','test']
special_file = ['sents.txt','sentiments.txt']

list_data = {}

def read_data(path):
    print(path)
    data = []
    with codecs.open(path, "r", "utf-8") as f:
        for item in f:
            data.append(item.replace('\n',''))
    return data


for folder in list_folder:
    sents_path = f'{source_path}/{folder}/{special_file[0]}'
    lb_path = f'{source_path}/{folder}/{special_file[1]}'
    a = read_data(sents_path)
    b = read_data(lb_path)
    data = [[x,y] for x,y in zip(a,b)]
    data = pd.DataFrame(data,columns=['sentence','label'])
    print(data)
    if folder == 'dev':
        folder = 'val'
    data.to_csv(f'{target_path}/final_{folder}.csv',index=False)

