# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 23:58:59 2019

@author: Raeed
"""
import pandas as pd
import collections
from collections import Counter
import matplotlib.pyplot as plt 

path = 'E:\Arabic_dataset_prepared/datasets statistics/'
result = pd.read_csv(path+ 'datasetsstatisticsshort.txt' ,  sep='|')
#        , index_col='Model_Name'


result.to_excel (path + 'datasetsstatisticsshort.xlsx', index = None, header=True)
   