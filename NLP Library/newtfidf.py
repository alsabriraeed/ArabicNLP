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

import os

def writingtofilewithindices(weighted_array,path,path1):
    print(len(weighted_array[0]))
    print(len(weighted_array))
#    f1 = open(path1,"w",encoding="utf8")
    with open(path1,"w",encoding="utf8") as f1, open(path,"w",encoding="utf8") as f :
        for item in weighted_array:
            
#            f1.write("%s\n" % " ".join(str(x) for x in np.asarray(np.nonzero(item))[0,:]))
#            f1.write("%s\n" % " ".join(str(x) for x in np.asarray(np.nonzero(item))[0,:]))
            indices = np.asarray(np.nonzero(item))
#            indices = np.asarray(np.nonzero(item))
#            str1 = " ".join(str(x) for x in indices[0,:])
            str1 = " ".join(str(ceil(x * 1000000) / 1000000.0) for x in indices[0,:])
            f1.write(str1 + "\n")
            f.write("%s\n" % " ".join(str( ceil(x * 1000000) / 1000000.0 ) for x in item[np.nonzero(item)]))
    f.close()
    f1.close()

def writingtofile(weighted_array,path):
#    print(len(weighted_array[0]))
    fileforclass = open(path,"w",encoding="utf8")
    for n in range(len(weighted_array)):
        stringweight =""
        for j in range(len(weighted_array[n])):
           stringweight += str(weighted_array[n][j]) + " "
 
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

def writingVocabulary(vocab_array,path ):
#    print(len(vocab_array))
    filevocab = open(path ,"w",encoding="utf8")
    stringweight =""
    for n in range(len(vocab_array)):
        stringweight += str(vocab_array[n] ) + " "
 
    filevocab.write(stringweight)
    filevocab.close()
    
    
def writingVocabularybyword(vocab,vocab2,path ):
    filevocab = open(path ,"w",encoding="utf8")
    for n in range(len(vocab)):
        filevocab.write(vocab[n] + ":    :"+ vocab2[n]+ "\n")
 
    filevocab.close()
    

def normalize(weighted_matrix5):
    max_value = max(weighted_matrix5)
    min_value = min(weighted_matrix5)
    
    for i in range(len(weighted_matrix5)):
        weighted_matrix5[i] = (weighted_matrix5[i] - min_value)/ (max_value -  min_value)
    
    return weighted_matrix5
    

def calculateLogwlda_entropy(logwldaweight, entropyweight):
    A=np.zeros(entropyweight.shape)
    print(entropyweight.shape)
    for i in range(entropyweight.shape[0]):
        for j in range(entropyweight.shape[1]):
            if entropyweight[i][j]== 0. :
                continue
            else:
                A[i][j] = entropyweight[i][j] + logwldaweight[j]
    return A
    

if __name__ == '__main__':
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    datasets =['Akhbarona'] #,'Akhbarona'
#    versions =["version9.1"]
    versions =["version4.1"]

#    "version2","version2.1","version3","version3.1",
#    models = ['tfidf','cew'] 
#    datasets =['ArabicAC']#"version2",//  improvedhammingtfidf
#    versions =['version7','version7.1']
#    versions =["version3.1"]
#    models = ['entropy']  
    models = ['newtfidf']#,'cew'//,'tfrstf'
#    datasettype='long' # 'long'
    datasettypes = ['short']# ,'short', 'long'
    for datasettype in datasettypes:
        for dataset in datasets:
            for version in versions:
                          
                print(datasettype+" "+dataset +" "+ version+" ok")
    #            path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/'
            #    path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/'
#                path = 'F:\Arabic dataset prepared\\'+dataset+r'\\'+version+'\\'+datasettype+'\/'
                path = '/home1/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
#                path = '/log05 sftp (172.18.1.179)/self/nuist/scratch/wangwen/wangwen_ali/raeed/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
    #            name = 'forpractice'
            #    name = 'ArabicCorpusACTwosentence1cleaned'
                name = dataset+'c'
    #            name =dataset
                with open(path + name + '.txt' , 'r',encoding="utf-8-sig") as f:
                    raw_docs = [line.replace("\n","") for  line in f]
                 
                vec = CountVectorizer()     
                X = vec.fit_transform(raw_docs)
            
                df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
            #    del X
                
#                vocabularies = df.columns
#                writingVocabulary(vocabularies,path + 'vocab.txt')
#            
                numpy_matrix = df.as_matrix()
                del df
    
            # tfidf new for the paper "Improved TFIDF in big news retrieval: An empirical study"
                for mod in range(len(models)):
                        
                    if (models[mod]=='newtfidf'):
                         # newtfidf    
                        weighted_matrix2 = weights.updatenewtfidf(numpy_matrix)
                
                        total_weight3=np.asarray(weighted_matrix2).sum(axis=0)
                        
                        writingTotalweighttofile(total_weight3,path +  'newtfidf.txt')
                        if (dataset !='ArabicAC' and datasettype =='long'):
                            writingtofilewithindices(weighted_matrix2,path + 'singlenewtfidf.txt',path + 'singlenewtfidfindices.txt')
                        elif (dataset =='ArabicAC' and datasettype =='long'):
                            writingtofile( weighted_matrix2,path + 'singlenewtfidf.txt')
                        