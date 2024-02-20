# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:37:31 2019

@author: raeed
"""

import weights
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from math import ceil
from nltk.tokenize import word_tokenize
import os
import warnings
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
        csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
def writingtofilewithindices(weighted_array,path,path1):
    print(len(weighted_array[0]))
    print(len(weighted_array))
#    f1 = open(path1,"w",encoding="utf8")
    with open(path1,"w",encoding="utf8") as f1, open(path,"w",encoding="utf8") as f :
        for item in weighted_array:
            
#            f1.write("%s\n" % " ".join(str(x) for x in np.asarray(np.nonzero(item))[0,:]))
#            f1.write("%s\n" % " ".join(str(x) for x in np.asarray(np.nonzero(item))[0,:]))
            indices = np.asarray(np.nonzero(item))
            f.write("%s\n" % " ".join(str( ceil(x * 1000000) / 1000000.0 ) for x in item[np.nonzero(item)]))
            f1.write("%s\n" % " ".join(str( ceil(x * 1000000) / 1000000.0 ) for x in indices[0,:])) 
  
    f.close()
    f1.close()

def writingtofile(weighted_array,path):
#    print(len(weighted_array[0]))
    fileforclass = open(path,"w",encoding="utf8")
    for n in range(len(weighted_array)):
        stringweight =""
        for j in range(len(weighted_array[n])):
#           stringweight += str(weighted_array[n][j]) + " "
           stringweight +=str( ceil(weighted_array[n][j] * 1000000) / 1000000.0 ) + " "
 
        fileforclass.write(stringweight +"\n")
    fileforclass.close()
    

   
def writingTotalweighttofile(weighted_array,filepath):
#    print(len(weighted_array))
    filefortotalweight = open(filepath,"w",encoding="utf8")
    stringweight =""
    for n in range(len(weighted_array)):
        stringweight += str( ceil(weighted_array[n] * 1000000) / 1000000.0 ) + " "
 
    filefortotalweight.write(stringweight +"\n")
    filefortotalweight.close()


    

    

def normalize(weighted_matrix5):
    max_value = max(weighted_matrix5)
    min_value = min(weighted_matrix5)
    
    for i in range(len(weighted_matrix5)):
        weighted_matrix5[i] = (weighted_matrix5[i] - min_value)/ (max_value -  min_value)
    
    return weighted_matrix5
    

    

if __name__ == '__main__':
    
    
    warnings.simplefilter('ignore',SparseEfficiencyWarning)
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
#    datasets =['AlKhaleej','AlArabiya','Akhbarona'] #,'Akhbarona'
    datasets =['ArabicAC']
#    versions =["version2.1","version3.1","version4.1","version5.1","version7.1"]
    versions =["version7.1"]

    models = ['pmi']#,'cew'//,'tfrstf'
#    datasettype='long' # 'long'
    datasettypes = [ 'long']# ,'short', 'long'
    for datasettype in datasettypes:
        for dataset in datasets:
            for version in versions:
                          
                print(datasettype+" "+dataset +" "+ version+" ok")
 
                path = 'F:\Arabic_dataset_prepared\\'+dataset+r'\\'+version+'\\'+datasettype+'\/'
   
                with open(path + 'log_WLDA.txt' , 'r',encoding="utf-8-sig") as f:
                    log_WLDA = [word_tokenize(line.replace("\n","")) for  line in f]             
                with open(path + 'entropy.txt' , 'r',encoding="utf-8-sig") as f:
                    entropy = [word_tokenize(line.replace("\n","")) for  line in f] 
                log_WLDA = np.asarray(log_WLDA[0],dtype='float32')
                entropy = np.asarray(entropy[0],dtype='float32')

                print(len(log_WLDA))
                print(len(entropy))
