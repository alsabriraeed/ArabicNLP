# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 23:58:59 2019

@author: Raeed
"""
import pandas as pd
import collections
from collections import Counter
import matplotlib.pyplot as plt 


#         datasets =['ArabicAC','AlKhaleej','AlArabiya','Akhbarona']
#         versions =["version2","version2.1","version3","version3.1","version4",
#				"version4.1","version5","version5.1","version6","version6.1","version7","version7.1"]
#datasets =['ArabicAC']
#versions =["version7.1"]
#datasettype='short' # 'long'
#for dataset in datasets:
#    for version in versions:
#        path = 'F:\Arabic dataset prepared\\'+dataset+'\\'+version+'\\'+datasettype+'\\'
#        path = '/home1/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
path = r'/home1/raeed/Arabic dataset prepared/datasets statistics/datasetsstatistics'
#        result = pd.read_csv(path + dataset+'_'+ version+'_'+datasettype +'.user',  sep='|')
result = pd.read_csv(path+'.txt',  sep='|')
#        , index_col='Model_Name'
print(result.head(3))


result.to_excel (path  +'.xlsx', index = None, header=True)
   