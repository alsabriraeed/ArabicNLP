# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 20:37:53 2020

@author: Raeed
"""

import pandas as pd
import numpy as np

#url = 'F:/ArabicDatasets/ArSenTD-LEV/ArSenTD-LEV.tsv'
url ='F:/ArabicDatasets/DataSet for Arabic Classification/arabic_dataset_classifiction.csv/arabic_dataset_classifiction.csv'
    
chipo = pd.read_csv(url, sep = '\t')
print(chipo.columns)
print(chipo.head(10))
#with open('F:/ArabicDatasets/ArSenTD-LEV/Tweet.txt','a',encoding="utf8") as f:
#    f.write(chipo.Tweet.to_string(header = False, index = False))
with open('F:/ArabicDatasets/DataSet for Arabic Classification/arabic_dataset_classifiction.csv/arabic_dataset_classifiction.txt','a',encoding="utf8") as f:
    f.write(chipo['text,targe'].to_string(header = False, index = False))