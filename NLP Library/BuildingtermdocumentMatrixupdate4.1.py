# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:37:31 2019

@author: raeed
"""

import weights
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import wordTowordcoocurrence
from math import ceil
from TF_RTF import termfrequency 
from scipy.spatial import distance
import OkapiBM25
import TF_RTF
import TFRFST

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
    
def writingTotalweighttofile_adding(weighted_array,weighted_array1):
    print(len(weighted_array))
    print(len(weighted_array1))
    filefortotalweight = open('F:\output/ArabicCorpusAClogentropytfidf.txt',"w",encoding="utf-8-sig")
    stringweight =""
    for n in range(len(weighted_array)):
        stringweight += str(weighted_array[n] + weighted_array1[n]) + " "
 
    filefortotalweight.write(stringweight +"\n")
    filefortotalweight.close()
  
def writingTotalweighte_adding(weighted_array, weighted_array1, weighted_array2 ):
#    print(len(weighted_array))
    filefortotalweight = open('F:\output/ArabicCorpusACWeightlogentropythree.txt',"w",encoding="utf8")
    stringweight =""
    for n in range(len(weighted_array)):
        stringweight += str(weighted_array[n] + weighted_array1[n]+weighted_array2[n]) + " "
 
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
#    datasets =['ArabicAC','AlKhaleej','AlArabiya','Akhbarona'] #,'Akhbarona'
    versions =["version2","version2.1","version3","version3.1","version4","version4.1","version5","version5.1","version6","version6.1","version7","version7.1"]
#    "version2","version2.1","version3","version3.1",
    models = ['improvedtfidf','improvedV2tfidf','improvedV2hammingtfidf','improvedV3hammingtfidf','newtfidf'
                    ,'tfidf','pmi','okapibm25','tfrtf','cew','tfrstf',''] 
    datasets =['Akhbarona']#"version2",
#    versions =['version7','version7.1']
#    versions =["version3.1"]
#    models = ['cew','pmi','okapibm25','tfrtf','tfrstf']  
#    models = ['tfrstf']#,'cew'//,'tfrstf'
#    datasettype='long' # 'long'
    datasettypes = ['long']# ,'short', 'long'
    for datasettype in datasettypes:
        for dataset in datasets:
            for version in versions:
                print(datasettype+" "+dataset +" "+ version+" ok")
    #            path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/'
            #    path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/'
#                path = 'F:\Arabic dataset prepared\\'+dataset+r'\\'+version+'\\'+datasettype+'\/'
#                path = '/home1/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
                path = '/log05 sftp (172.18.1.179)/self/nuist/scratch/wangwen/wangwen_ali/raeed/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
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
                
                vocabularies = df.columns
                writingVocabulary(vocabularies,path + 'vocab.txt')
            
                numpy_matrix = df.as_matrix()
                del df
    
            # tfidf new for the paper "Improved TFIDF in big news retrieval: An empirical study"
                for mod in range(len(models)):
                    if (models[mod]=='improvedtfidf'):
                        
                        D = euclidean_distances(X)
                        B = weights.binary(numpy_matrix)
                        B = B.toarray()
                        R = D.dot(B)
                        weighted_matrix2 = weights.improvednewtfidf(numpy_matrix, R)
                    
                        total_weight3=np.asarray(weighted_matrix2).sum(axis=0)
                       
                        writingTotalweighttofile(total_weight3,path +  'improvedtfidf.txt')
                        writingtofile( weighted_matrix2,path + 'singleimprovedtfidf.txt')
                    
                    elif (models[mod]=='improvedV2tfidf'):
                        
                        ## tfidf new for the paper "Improved TFIDF in big news retrieval: An empirical study" nersion2+euclidean
                        tfidf  = weights.newtfidf(numpy_matrix)
                        tfidf = np.asarray(tfidf)
                    
                        D = euclidean_distances(tfidf)
                        B = weights.binary(tfidf)
                        B = B
                        R = D.dot(B)
                        weighted_matrix2 = weights.improvednewtfidf(tfidf, R)
                    
                        total_weight3=np.asarray(weighted_matrix2).sum(axis=0)
                        
                        writingTotalweighttofile(total_weight3,path +  'improvedV2tfidf.txt')
                        writingtofile( weighted_matrix2,path + 'singleimprovedV2tfidf.txt')
                    elif (models[mod]=='improvedV2hammingtfidf'):
                        # tfidf new for the paper "Improved TFIDF in big news retrieval: An empirical study" nersion2+ using hamming distance
                        tfidf  = weights.newtfidf(numpy_matrix)
                        tfidf = np.asarray(tfidf)
                    
                        D = weights.hammingpairwise_distance(tfidf)
                        D = np.asarray(D)
                    
                        B = weights.binary(tfidf)
                        B = B
                        R = D.dot(B)
                        weighted_matrix2 = weights.improvednewtfidf(tfidf, R)
                            
                        total_weight3=np.asarray(weighted_matrix2).sum(axis=0)
                        
                        writingTotalweighttofile(total_weight3,path +  'improvedV2hammingtfidf.txt')
                        writingtofile( weighted_matrix2,path + 'singleimprovedV2hammingtfidf.txt')
                    elif (models[mod]=='improvedV3hammingtfidf'):
                        #tfidf new for the paper "Improved TFIDF in big news retrieval: An empirical study" nersion3* using hamming distance
                        D = euclidean_distances(X)
                        B = weights.binary(numpy_matrix)
                        B = B.toarray()
                        R = D.dot(B)
                        weighted_matrix2 = weights.improvednewtfidf(numpy_matrix, R)
                                
                        D = weights.hammingpairwise_distance(weighted_matrix2)
                        D = np.asarray(D)
            
                        B = weights.binary(weighted_matrix2)
                        B = B
                        R = D.dot(B)
                        weighted_matrix2 = weights.improvednewtfidf(weighted_matrix2, R)
                           
                        total_weight3=np.asarray(weighted_matrix2).sum(axis=0)
            
                        writingTotalweighttofile(total_weight3,path +  'improvedV3hammingtfidf.txt')
                        writingtofile( weighted_matrix2,path + 'singleimprovedV3hammingtfidf.txt')
                        
                    elif (models[mod]=='newtfidf'):
                         # newtfidf    
                        weighted_matrix2 = weights.newtfidf(numpy_matrix)
                
                        total_weight3=np.asarray(weighted_matrix2).sum(axis=0)
                        
                        writingTotalweighttofile(total_weight3,path +  'newtfidf.txt')
                        writingtofile( weighted_matrix2,path + 'singlenewtfidf.txt')
                        
                    elif (models[mod]=='tfidf'):
                        # tfidf    
                        weighted_matrix2 = weights.tfidf(numpy_matrix)
                    
                        weighted_array2 = weighted_matrix2.toarray()
                        total_weight3=np.asarray(weighted_array2).sum(axis=0)
                        
                        writingTotalweighttofile(total_weight3,path +  'tfidf.txt')
                        writingtofile( weighted_array2,path + 'singletfidf.txt')
            
                    elif (models[mod]=='cew'):
                        ## CEW--------------------------------------------
                        weighted_matrix5 = weights.log_WLDA(numpy_matrix)
                        
                        writingTotalweighttofile(weighted_matrix5,path +'log_WLDA.txt')
                        weighted_matrix5 = normalize(weighted_matrix5)
                        
                    
                        weighted_matrix6 = wordTowordcoocurrence.main(raw_docs)
                        writingTotalweighttofile(weighted_matrix6,path+ 'entropy.txt')
                        weighted_matrix6 = normalize(weighted_matrix6)
                        logldaentropyweight = np.multiply(weighted_matrix5,weighted_matrix6)
                        
                        writingTotalweighttofile(logldaentropyweight,path+ 'CEW.txt')
                        
                    elif (models[mod]=='pmi'):
                        # PMI ========================================================
                        weighted_matrix2 = weights.pmi(numpy_matrix)
                            
                        weighted_array2 = weighted_matrix2.toarray()
                        total_weight3=np.asarray(weighted_array2).sum(axis=0)
                       
                        writingTotalweighttofile(total_weight3, path+'pmi.txt')
                        writingtofile( weighted_array2, path+'singlepmi.txt')
                        
                    elif (models[mod]=='entropy'):
                        #logentropy2 ---------------------------------------------------------
                        weighted_matrix2 = weights.logentropy2(numpy_matrix)
                    
                        weighted_array2 = weighted_matrix2.toarray()
                        total_weight3=np.asarray(weighted_array2).sum(axis=0)
                        
                        writingTotalweighttofile(total_weight3, path+'entropy.txt')
                        writingtofile( weighted_array2, path+'singlentropy.txt')
            
                    elif (models[mod]=='okapibm25'):
                        obj = OkapiBM25.main(path,name)
                    elif (models[mod]=='tfrtf'):
                        obj = TF_RTF.main(path,name)
                    elif (models[mod]=='tfrstf'):
                        obj = TFRFST.main(path,name,version)
                        
            
            #okapibm25
            #        del numpy_matrix
            #        del vocabularies
            #        del raw_docs
    
    
    
    
