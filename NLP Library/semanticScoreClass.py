# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:35:29 2019

@author: Raeed
"""
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import re
import sys
#sys.path.append(r'/home1/raeed/process-arabic-text-master')
#sys.path.append(r'/home1/raeed/word2vecmodel')
sys.path.append(r'/log05 sftp (172.18.1.179)/self/nuist/scratch/wangwen/wangwen_ali/raeed/raeed/process-arabic-text-master')
sys.path.append(r'/log05 sftp (172.18.1.179)/self/nuist/scratch/wangwen/wangwen_ali/raeed/raeed/word2vecmodel')
warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec
#model = gensim.models.Word2Vec.load('Wiki-SG')
class semantic_scores:
    
    
    def semantic_score_matrix(doc,version,model):
            model = gensim.models.Word2Vec.load(r'/home1/raeed/word2vecmodel/'+model+'.model')
#            model = gensim.models.Word2Vec.load(r'/log05 sftp (172.18.1.179)/self/nuist/u/home/wangwen/wangwen_ali/raeed/word2vecmodel/'+version+'.model')
#        with open(r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/forpractice.txt', 'r',encoding="utf-8-sig") as f:
#            raw_docs = [line.replace("\n", "") for line in f]
        #data = []
        #for doc in raw_docs:
        #    temp = []
        #    for j in word_tokenize(doc):
        #        temp.append(j)
        #    data.append(temp)
        
#        for doc in raw_docs:
            semantic_scores = []
#            tokens = word_tokenize(doc)    
            for token1 in doc:
                semantic_score = []
                for token2 in doc:
                    if token1 == token2:
                        semantic_score.append(0)
                    else:
                        semantic_score.append(model.similarity(token2,token1))
                semantic_scores.append(semantic_score)
            return(semantic_scores)
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
    #model2 = gensim.models.Word2Vec(data, min_count=1, size=100,
    #                                window=10, sg=1)
    
    
    # Print results
#    print("semantic similarity: ",
#          model.similarity('مطار', 'مخاطر'))
    
    # print("Cosine similarity between 'alice' " +
    #       "and 'machines' - Skip Gram : ",
    #       model2.similarity('alice', 'machines'))
