# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:35:29 2019

@author: Raeed
"""
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

import sys
#sys.path.append(r'F:\for master\MS CS\relating to research\new papers\more related to my work\process-arabic-text-master')
sys.path.append(r'/home1/raeed/process-arabic-text-master')
warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec
#model = gensim.models.Word2Vec.load('Wiki-SG')
class semantic_scores:
    
    
        versions =["version2","version3","version4",
				"version5","version6","version7"]
        for ver in versions:
                
    #        with open(r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/ArabicCorpusACTwosentence1cleaned.txt', 'r',encoding="utf-8-sig") as f:
            with open(r'/home1/raeed/Arabic dataset prepared/wiki/'+ver+'/wiki.txt', 'r',encoding="utf-8-sig") as f:
                raw_docs = [line.replace("\n", "") for line in f]
            data = []
            for doc in raw_docs:
                temp = []
                for j in word_tokenize(doc):
                    temp.append(j)
                data.append(temp)
            
    
        #for doc in data:
        #    for word in range(len(doc)- 2):
        #        for word+1 in range()
        
        # Create CBOW model
        # model1 = gensim.models.Word2Vec(data, min_count=1,
        #                               size=100, window=5)
        
        # Print results
        # print("Cosine similarity between 'alice' " +
        #       "and 'wonderland' - CBOW : ",
        #       model1.similarity('اصابه', 'محمد'))
        
        # print("Cosine similarity between 'alice' " +
        #       "and 'machines' - CBOW : ",
        #       model1.similarity('alice', 'machines'))
        
        # Create Skip Gram model
            model2 = gensim.models.Word2Vec(data, min_count=1, size=100,
                                        window=10, sg=1)
        
        
        # Print results
    #    print("semantic similarity: ",
    #          model.similarity('مطار', 'مخاطر'))
            model2.save(r'/home1/raeed/word2vecmodel/'+ver+'.model')
    #        model3 = Word2Vec.load(r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/forpractice.model')
    #        print("Cosine similarity between 'alice' " +
    #               "and 'machines' - Skip Gram : ",
    #               model3.similarity('شركه', 'مطار'))
