import os
import time
import pandas as pd
import numpy as np
import re
import jieba
from jieba import analyse
from collections import Counter

data = pd.DataFrame()
file_folder = "./data/"
for file in os.listdir(file_folder):
    file_path = os.path.join(file_folder, file)
    f = open(file_path, encoding = "utf-8")
    content = f.read()
    stock = file[0:6]
    df = pd.DataFrame([[stock, file, content]], columns=["stock","filename","content"])
    data = pd.concat([data, df])
    f.close()

data = data.reset_index(drop = True)

from cnsenti import Sentiment

senti = Sentiment(pos='./dict/formal_pos.txt',  
                  neg='./dict/formal_neg.txt',  
                  merge=False,             
                  encoding='utf-8') 

data.insert(data.shape[1], 'words', 0)
data.insert(data.shape[1], 'sentences', 0)
data.insert(data.shape[1], 'pos', 0)
data.insert(data.shape[1], 'neg', 0)    

a = 0
for content in data["content"]:
    result = senti.sentiment_count(content)
    data.iloc[a,3] = result["words"] 
    data.iloc[a,4] = result["sentences"]
    data.iloc[a,5] = result["pos"]  
    data.iloc[a,6] = result["neg"]
    a = a + 1 

simple = data[["stock","words","sentences","pos","neg"]]
simple.to_csv("./dict/simple.csv")