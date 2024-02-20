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
versions =["version2.1","version3.1",
				"version4.1","version5.1","version7.1"]
datasets =['ArabicAC','AlKhaleej','AlArabiya','Akhbarona']
#datasets =['ArabicAC']
datasettype='long' # 'long'
for dataset in datasets:
    for version in versions:
        path = '/nuist/scratch/wangwen/wangwen_ali/raeed/Arabic_dataset_prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
        path1 = '/nuist/scratch/wangwen/wangwen_ali/raeed/Arabic_dataset_prepared/results/'
#        path = 'F:\Arabic dataset prepared\\'+dataset+'\\'+version+'\\'+datasettype+'\\'
#        path = '/home1/raeed/Arabic dataset prepared/'+dataset+'/'+version+'/'+datasettype+'/'
        result = pd.read_csv(path + dataset+'_'+ version+'_'+datasettype +'.txt',  sep='|')
#        , index_col='Model_Name'
#        print(result.head(20))
#        print(result.shape[0])
        print(result.Accuracy.max())
        result.to_excel (path1 + dataset+'_'+ version+'_'+datasettype +'.xlsx', index = None, header=True)
       