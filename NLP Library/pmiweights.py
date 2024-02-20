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
    datasets =['AlKhaleej']
#    versions =["version2.1","version3.1","version4.1","version5.1","version7.1"]
    versions =["version2.1"]
#    versions =["version7.1"]

#    "version2","version2.1","version3","version3.1",
#    models =   ['tfidf','cew'] 
#    datasets =['ArabicAC']#"version2",//  improvedhammingtfidf
#    versions =['version7','version7.1']
#    versions =["version3.1"]
#    models = ['entropy']  
    models = ['pmi']#,'cew'//,'tfrstf'
#    datasettype='long' # 'long'
    datasettypes = [ 'long']# ,'short', 'long'
    for datasettype in datasettypes:
        for dataset in datasets:
            for version in versions:
                          
                print(datasettype+" "+dataset +" "+ version+" ok")
    #            path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/'
            #    path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/'
#                path = 'F:\Arabic_dataset_prepared\\'+dataset+r'\\'+version+'\\'+datasettype+'\/'
#                path = '/home1/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
                path = '/nuist/scratch/wangwen/wangwen_ali/raeed/Arabic_dataset_prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
    #            name = 'forpractice'
            #    name = 'ArabicCorpusACTwosentence1cleaned'
                name = dataset+'c'
    #            name =dataset
                with open(path + name + '.txt' , 'r',encoding="utf-8-sig") as f:
                    raw_docs = [line.replace("\n","") for  line in f]
                    
                with open(path + 'log_WLDA20.txt' , 'r',encoding="utf-8-sig") as f:
                    log_WLDA = [word_tokenize(line.replace("\n","")) for  line in f]             
                
                log_WLDA = np.asarray(log_WLDA[0],dtype='float32')
#                print(log_WLDA)
                vec = CountVectorizer()     
                X = vec.fit_transform(raw_docs)
                
                df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
                del X
                del vec
                
#                vocabularies = df.columns
#                writingVocabulary(vocabularies,path + 'vocab.txt')
#            
                numpy_matrix = df.as_matrix()
                del df
    
            # tfidf new for the paper "Improved TFIDF in big news retrieval: An empirical study"
                for mod in range(len(models)):
                        
                    if (models[mod]=='pmi'):
                         # newtfidf    
#                        weighted_matrix2 = weights.pmi(numpy_matrix)
                        weighted_matrix2 = weights.pmi_log_WLDA(numpy_matrix,log_WLDA)
#                        print(type(weighted_matrix2))
#                        print(weighted_matrix2.toarray())
                        weighted_matrix2 =np.array(weighted_matrix2)
                        weighted_matrix2[np.isinf(weighted_matrix2)]=0
#                        print(weighted_matrix2)
#                        print(type(weighted_matrix2))
#                        weighted_matrix2 =weighted_matrix2.toarray()
#                        where_are_NaNs = np.isnan(weighted_matrix2)
#                        weighted_matrix2[where_are_NaNs] = 0
#                                                
#                        weighted_matrix2= np.where(weighted_matrix2==-np.inf, 0, weighted_matrix2) 
#                        weighted_matrix2= np.where(weighted_matrix2==np.nan, 0, weighted_matrix2) 
#                        print(weighted_matrix2.shape)
#                        print(weighted_matrix2[0][0])
#                        weighted_matrix2 = weights.updatenewtfidf(numpy_matrix)
                
                        total_weight3=np.asarray(weighted_matrix2).sum(axis=0)
                        
#                        total_weight3= np.where(total_weight3==-np.inf, 0, total_weight3) 
#                        total_weight3= np.where(total_weight3==np.nan, 0, total_weight3)
#                        total_weight3 = total_weight3.toarray()
#                        print(type(total_weight3))
#                        print("total_weight3\n",total_weight3)
                        if (datasettype =='short'):
                            writingTotalweighttofile(total_weight3,path +  'pmilogWLDA.txt')
                        elif (datasettype =='long'):
                            writingtofilewithindices(weighted_matrix2,path + 'singlepmi20.txt',path + 'pmilogWLDA20indices.txt')
#                        elif (dataset =='ArabicAC' and datasettype =='long'):
#                            writingtofile( weighted_matrix2,path + 'singlepmilogWLDA.txt')
#                        
