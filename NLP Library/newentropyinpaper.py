# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:22:10 2019

@author: Raeed
"""
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from math import ceil
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import create_co_occurences_matrix as coocurwords
#from numba import cuda
#@cuda.jit
def normalize(weighted_matrix5):
    max_value = max(weighted_matrix5)
    min_value = min(weighted_matrix5)
    
    for i in range(len(weighted_matrix5)):
        weighted_matrix5[i] = (weighted_matrix5[i] - min_value)/ (max_value -  min_value)
    
    return weighted_matrix5
    

def writingVocabulary(vocab_array,path ):
#    print(len(vocab_array))
    filevocab = open(path ,"w",encoding="utf8")
    stringweight =""
    for n in range(len(vocab_array)):
        stringweight += str(vocab_array[n] ) + " "
 
    filevocab.write(stringweight)
    filevocab.close()
    
def writingtofile1(weighted_array,path,path1):
    print(len(weighted_array[0]))
    print(len(weighted_array))
#    f1 = open(path1,"w",encoding="utf8")
    with open(path1,"w",encoding="utf8") as f1, open(path,"w",encoding="utf8") as f :
        c =0
        for item in weighted_array:
            
#            f1.write("%s\n" % " ".join(str(x) for x in np.asarray(np.nonzero(item))[0,:]))
#            f1.write("%s\n" % " ".join(str(x) for x in np.asarray(np.nonzero(item))[0,:]))
            indices = np.asarray(np.nonzero(item))
            f.write("%s\n" % " ".join(str(x) for x in item[np.nonzero(item)]))
            f1.write("%s\n" % " ".join(str(x) for x in indices[0,:]))
#            print(item)
#            print(c)
#            stre = "%s\n" % " ".join(str(x) for x in np.asarray(np.nonzero(item))[0,:])
#            c+=1
#            print(type(np.asarray( np.nonzero(item))))
#            f1.write("%s\n" % " ".join(str(x) for x in np.asarray(np.nonzero(item))[0,:]))   
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
    

if __name__ == '__main__':
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    datasets =['AlArabiya','Akhbarona','AlKhaleej'] #,'AlKhaleej','AlArabiya','Akhbarona'
    versions =["version2.1"] #,"version3.1","version4.1","version5.1","version7.1"
#    "version2","version2.1","version3","version3.1",//"version2.1","version3.1",

#    datasets =['ArabicAC']#"version2",

#    models = ['cew','pmi','okapibm25','tfrtf','tfrstf']  
#    models = ['tfrstf']#,'cew'//,'tfrstf'
#    datasettypes=['short'] # 'long'
    datasettypes = ['short']# ,'short', 'long'
#    datasettypes = ['long']
    for datasettype in datasettypes:
        for dataset in datasets:
            for version in versions:
                print(datasettype+" "+dataset +" "+ version+" ok")
    #            path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/'
            #    path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/'
#                path = 'F:\Arabic_dataset_prepared\\'+dataset+r'\\'+version+'\\'+datasettype+'\/'
#                path = '/content/drive/My Drive/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
#                path = '/home1/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
                path = '/nuist/scratch/wangwen/wangwen_ali/raeed/Arabic_dataset_prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
    #            name = 'forpractice'
            #    name = 'ArabicCorpusACTwosentence1cleaned'
                name = dataset+'c'
    #            name =dataset
                with open(path + name + '.txt' , 'r',encoding="utf-8-sig") as f:
                    raw_docs = [line.replace("\n","") for  line in f]
                 
#                vectorizer = TfidfVectorizer(encoding='utf-8',sublinear_tf=True)     
#                X = vectorizer.fit_transform(raw_docs)
#                weighted_array2 = X.toarray()
#                total_weight3=np.asarray(weighted_array2).sum(axis=0)
#                print(X.shape)
#                print(total_weight3.shape)
#                writingTotalweighttofile(total_weight3,path +  'pythontfidf.txt')
                
                vec = CountVectorizer()     
                X = vec.fit_transform(raw_docs)
                X = X.astype(np.int32)
                df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names(), dtype='int32')
#                vocabularies = df.columns
                
#                numpy_matrix = df.values
                del X
#                del numpy_matrix
                
                vocabularies = df.columns
                writingVocabulary(vocabularies,path + 'vocab.txt')
                del df
#                weighted_matrix2 = weights.tfidf(X)   
#                writingVocabulary(vocabularies,path + 'vocab.txt')
##                print(weighted_matrix2.toarray())
##                print("weighted_matrix2 ", weighted_matrix2.shape )                
#                weighted_array2 = weighted_matrix2.toarray()
##                writingTotalweighttofile( np.count_nonzero(weighted_array2, axis=1),path + 'countword.txt')
#
#                total_weight3=np.asarray(weighted_array2).sum(axis=0)
##                print(X.shape)
##                print(total_weight3.shape)
#                writingTotalweighttofile(total_weight3,path +  'tfidf.txt')
#                writingtofile1( weighted_array2,path + 'singletfidf.txt',path + 'singletfidfindices.txt')
#                weighted_matrix5 = weights.log_WLDA(numpy_matrix)
#                        
#                writingTotalweighttofile(weighted_matrix5,path +'log_WLDA.txt')
                
                  ## CEW--------------------------------------------
#                weighted_matrix5 = weights.log_WLDA(numpy_matrix)                        
#                writingTotalweighttofile(weighted_matrix5,path +'log_WLDA.txt')
#                weighted_matrix5 = normalize(weighted_matrix5)
#                print(raw_docs)
                
#                print(wtwoccur.calculatewordtowordcoocurrence1(raw_docs))
                raw_docs = [word_tokenize(sen) for sen in raw_docs]
                print(len(vocabularies))
                raw_docs, word_to_id = coocurwords.create_co_occurences_matrix(vocabularies, raw_docs)
                del vocabularies
                del word_to_id
                print(raw_docs.shape)
#                print(words_cooc_matrix.toarray())
                raw_docs = coocurwords.calprocoocurenofwords(raw_docs.astype(np.float32))
#                print(calprocoocurenofwords1)
                print(raw_docs.shape)
                print("heeeeeeeeeeeeeeeeeeeeeereee")
                raw_docs = coocurwords.caculateentropy(pd.DataFrame( raw_docs,dtype=np.float32))
#                print(caculatedentropy)
                writingTotalweighttofile(raw_docs,path+ 'entropy.txt')
#                caculatedentropy = normalize(caculatedentropy)
#                logldaentropyweight = np.multiply(weighted_matrix5,caculatedentropy)
                        
#                writingTotalweighttofile(logldaentropyweight,path+ 'CEW.txt')
          
                
                