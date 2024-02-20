# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:37:31 2019

@author: raeed
"""


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
#import os

if __name__ == '__main__':
#    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
#    datasets =['ArabicAC','AlKhaleej','AlArabiya','Akhbarona']
    datasets =['Khaleejshort']

#    versions =["version2.1","version3.1",
#				"version4.1","version5.1","version7.1"]

    versions =['version1']

    datasettypes = ['short']# ,'short' 'long',
    
    strstatistics = "dataset" +"|"+ "version" +"|"+ "datasettype"+"|" + "uniqueterms"  + "|" + "numberofterm"+"|"+ "avglengthofdocument" 

    datasetsstatistics = open(r'F:\ArabicDatasets\Khaleejshort\version1\short/datasetsstatisticsshort'+'.txt',"w",encoding="utf8")
#    datasetsstatistics = open(r'/home1/raeed/Arabic dataset prepared/datasets statistics/datasetsstatisticsshort.txt',"w",encoding="utf8")
#    datasetsstatistics = open(r'/log05 sftp (172.18.1.179)/self/nuist/scratch/wangwen/wangwen_ali/raeed/raeed/Arabic dataset prepared/datasets statistics/datasetsstatisticsshort.txt',"w",encoding="utf8")

    datasetsstatistics.write(strstatistics +"\n")
    for datasettype in datasettypes:
        for dataset in datasets:
            for version in versions:
                print(datasettype,"  ",dataset,"  ",version)
                 
    #            path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/'
            #    path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/'
                path = 'F:\ArabicDatasets\\'+dataset+r'\\'+version+'\\'+datasettype+'\/'
#                path = '/home1/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
#                path = '/log05 sftp (172.18.1.179)/self/nuist/scratch/wangwen/wangwen_ali/raeed/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
    #            name = 'forpractice'
            #    name = 'ArabicCorpusACTwosentence1cleaned'
#                name = dataset[3:] +'1'
                name = 'Khaleejc'
    #            name =dataset
                raw_docs = []
                avglengthofdocument = 0
                numberofterm = 0 
                with open(path + name + '.txt' , 'r',encoding="utf-8-sig") as f:
                    text = [word_tokenize(line.replace("\n","")) for  line in f]
                    
                for  doc in text:
#                    raw_docs.append(line.replace("\n",""))
#                    avglengthofdocument += len(word_tokenize(line))
#                    print(len(doc))
                    numberofterm += len(doc)
#                print(numberofterm)
#                    raw_docs = [line.replace("\n","") for  line in f]
                avglengthofdocument = numberofterm/len(text) 
#                print(avglengthofdocument)
                uniqueterms = Dictionary(text)
                
#                print(len(uniqueterms))
                strstatistics = dataset +"|"+ version +"|"+ datasettype+"|" + str(len(uniqueterms))  + "|" + str(numberofterm)+"|"+ str(round(avglengthofdocument))  
                datasetsstatistics.write(strstatistics +"\n")
    datasetsstatistics.close()
     
    
            
          