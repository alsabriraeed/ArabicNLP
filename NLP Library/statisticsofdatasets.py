# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:37:31 2019

@author: raeed
"""


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
#import os

if __name__ == '__main__':
#    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
#    datasets =['ArabicAC','AlKhaleej','AlArabiya','Akhbarona']
    datasets =['Akhbarona']
#    versions =["version2","version2.1","version3","version3.1","version4",
#				"version4.1","version5","version5.1","version6","version6.1","version7","version7.1"]
    versions =["version2.1","version3.1",
				"version4.1","version5.1","version7.1"]
#    models = ['improvedtfidf','improvedV2tfidf','improvedV2hammingtfidf','improvedV3hammingtfidf','newtfidf'
#                    ,'tfidf','cew','pmi','okapibm25','tfrtf','tfrstf','','','','']  
#    datasets =['ArabicAC']
#    versions =['version5.1']
#    versions =["version4"]
#    models = ['cew','pmi','okapibm25','tfrtf','tfrstf']  
#    models = ['tfidf']
#    datasettype='long' # 'long'
    datasettypes = ['long']# ,'short' 'long',
    
    strstatistics = "dataset" +"|"+ "version" +"|"+ "datasettype"+"|" + "uniqueterms"  + "|" + "numberofterm"+"|"+ "avglengthofdocument" 

    datasetsstatistics = open(r'F:\Arabic dataset prepared\datasets statistics/datasetsstatistics'+'Ackbarona',"w",encoding="utf8")
#    datasetsstatistics = open(r'/home1/raeed/Arabic dataset prepared/datasets statistics/datasetsstatisticsshort.txt',"w",encoding="utf8")
#    datasetsstatistics = open(r'/log05 sftp (172.18.1.179)/self/nuist/scratch/wangwen/wangwen_ali/raeed/raeed/Arabic dataset prepared/datasets statistics/datasetsstatisticsshort.txt',"w",encoding="utf8")

    datasetsstatistics.write(strstatistics +"\n")
    for datasettype in datasettypes:
        for dataset in datasets:
            for version in versions:
                print(datasettype,"  ",dataset,"  ",version)
                 
    #            path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/'
            #    path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/'
                path = 'F:\Arabic dataset prepared\\'+dataset+r'\\'+version+'\\'+datasettype+'\/'
#                path = '/home1/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
#                path = '/log05 sftp (172.18.1.179)/self/nuist/scratch/wangwen/wangwen_ali/raeed/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
    #            name = 'forpractice'
            #    name = 'ArabicCorpusACTwosentence1cleaned'
                name = dataset+'c'
    #            name =dataset
                raw_docs = []
                avglengthofdocument = 0
                numberofterm = 0 
                with open(path + name + '.txt' , 'r',encoding="utf-8-sig") as f:
                    for  line in f:
                        raw_docs.append(line.replace("\n",""))
                        avglengthofdocument += len(word_tokenize(line))
                        
                        numberofterm += len(word_tokenize(line))
                    
#                    raw_docs = [line.replace("\n","") for  line in f]
                avglengthofdocument = avglengthofdocument/len(raw_docs) 
                vec = CountVectorizer()     
                X = vec.fit_transform(raw_docs)
            
                df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
            #    del X
                
                vocabularies = df.columns
                uniqueterms = len(vocabularies)
                
            
                numpy_matrix = df.as_matrix()
                del df
                
                strstatistics = dataset +"|"+ version +"|"+ datasettype+"|" + str(uniqueterms)  + "|" + str(numberofterm)+"|"+ str(round(avglengthofdocument))  
                datasetsstatistics.write(strstatistics +"\n")
    datasetsstatistics.close()
     
    
            
          