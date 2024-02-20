# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:52:23 2019

@author: Raeed
"""
import numpy as np
import pandas as pd
import math
from pylab import *
from scipy.sparse import * 
from sklearn.feature_extraction.text import CountVectorizer
from  nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import word_tokenize
from itertools import combinations
from collections import Counter
import BuildingtermdocumentMatrix as bmx



def calculatewordtowordcoocurrence1(sentences): 
    
#    print(sentences)
    vec = CountVectorizer()     
    X = vec.fit_transform(sentences)
    vocab = vec.get_feature_names()
#    vocab = list(set(word_tokenize(' '.join(sentences))))
#    for i in range(len(vocab)):
#        vocab[i] = vocab[i].replace(" ","")
    vocab.sort()
#    bmx.writingVocabularybyword(vocab,vocab,r'F:\output/comparingvocab1.txt')
#    for i in range(len(vocab)):
#        BuildingtermdocumentMatrix.writingVocabularybyword("  ",vocab[i])
#    print("vocabhavebeenwritten")
    
#    vocab = vocabularies
#    vocab = set(vocabularies)
#    vocab = list(vocab)
#    vocab.sort()
#    print("vocab :" ,vocab)
#    vocab = set(vocab)
#    print('Vocabulary:\n',vocab,'\n')
    
    token_sent_list = [word_tokenize(sen) for sen in sentences]
#    tokens = set()
#    for i in range(len(token_sent_list)):
#        for j in range(len(token_sent_list[i])):
#            tokens.add(token_sent_list[i][j])
#    vocab = list(tokens)
#    vocab.sort()
    for i in range(len(token_sent_list)):
        for j in range(len(token_sent_list[i])):
            if(token_sent_list[i][j]=='ن') or (token_sent_list[i][j]=='و'):
                print("doc : "+ str( i ), " word : " +str(j))
            
    
#    print('Each sentence in token form:\n',token_sent_list,'\n')

# Get the index of elements with value 15
#    result = np.where(token_sent_list == 'ي')
#    print(result)
    co_occ = {ii:Counter({jj:0 for jj in vocab if jj!=ii}) for ii in vocab}
#    print(co_occ['ي'])
    k=10
    
    for sen in token_sent_list:
        for ii in range(len(sen)):
            if ii < k:
                c = Counter(sen[0:ii+k+1])
                del c[sen[ii]]
                co_occ[sen[ii]] = co_occ[sen[ii]] + c
            elif ii > len(sen)-(k+1):
                c = Counter(sen[ii-k::])
                del c[sen[ii]]
                co_occ[sen[ii]] = co_occ[sen[ii]] + c
            else:
                c = Counter(sen[ii-k:ii+k+1])
                del c[sen[ii]]
                co_occ[sen[ii]] = co_occ[sen[ii]] + c
    
    # Having final matrix in dict form lets you convert it to different python data structures
    co_occ = {ii:dict(co_occ[ii]) for ii in vocab}
   
    return(pd.DataFrame.from_dict(co_occ))

def calprocoocurenofwords(wTowcoocurence):
    print("wTowcoocurence: ",type(wTowcoocurence))
    sumcoocrence = wTowcoocurence.sum(axis=1)
    wTowcoocurence1 = wTowcoocurence
    wTowcoocurence = wTowcoocurence.replace(np.nan, 0)
    for row in range(wTowcoocurence1.shape[1]):
        for col in range(wTowcoocurence1.shape[1]):
#            pass
            wTowcoocurence1.iat[row,col] = wTowcoocurence1.iat[row,col] / sumcoocrence[row]
    return wTowcoocurence1.replace(np.nan, 0)

#    print(sumcoocrence)
#    print(wTowcoocurence.shape)
#    print(sumcoocrence[0])
#    print(type(sumcoocrence))

def caculateentropy(procoocurenofwords):
#    procoocurenofwords1 = pd.DataFrame(index=procoocurenofwords.index)
    procoocurenofwords1 = procoocurenofwords.iloc[:,0]
#    print(" procoocurenofwords1 : ",procoocurenofwords1)
    for row in range(procoocurenofwords.shape[1]):
        sumforeachword = 0.0
        count = 0
        for col in range(procoocurenofwords.shape[1]):
            if procoocurenofwords.iat[row,col]== 0:
                continue
            count+=1
            sumforeachword += procoocurenofwords.iat[row,col] * log( procoocurenofwords.iat[row,col])
        normalizedentropy = -1 * ((-1 * sumforeachword)/abs(log(count)))
#        entropyweight.append(-1 * normalizedentropy)
#        print((-1 * normalizedentropy))
        procoocurenofwords1.iloc[row] = (-1 * round_(normalizedentropy,decimals = 9))
    
    return procoocurenofwords1

def main(ctxs):
#    ctxs = [
#            'krayyem like candy',
#            'krayyem plays candy',
#            'krayyem do not invite',
#            'krayyem is smart',
#            'smart is krayyem',        
#        ]
    
    #wTowcoocurence= calculatewordtowordcoocurrence(ctxs)
    wTowcoocurence1= calculatewordtowordcoocurrence1(ctxs)
    
    print(wTowcoocurence1)
    procoocurenofwords = calprocoocurenofwords(wTowcoocurence1)
    print(procoocurenofwords)
    entropybasedweight= caculateentropy(procoocurenofwords)
#    print("entropybasedweight: ",entropybasedweight )
    print("weighted_matrix6" ,entropybasedweight)
    return entropybasedweight
    #print(wTowcoocurence)
    #print(wTowcoocurence1)
    #print(type(wTowcoocurence))

#main(ctxs)