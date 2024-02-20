# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:37:31 2019

@author: raeed
"""

import weights
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.stats as ss
from pylab import *
import math
import BuildingtermdocumentMatrix as bmx
from math import ceil
import semanticScoreClass as ssc
import clustering_fun as func
def termfrequency(A):
    """
    Input: term: Term in the Document, doc: Document
    Return: Normalized tf: Number of times term occurs
      in document/Total number of terms in the document
    """
    lendoc = np.count_nonzero(A, axis=1)
    # Number of times the term occurs in the document
    tf = []
    for i in range(len(A)):
        x = []
        for j in range(len(A[i])):
            x.append(A[i][j] / lendoc[i])
        tf.append(x)

    return tf


def TFRSTF(A, Ranking, doc_length , unique_terms):
    RTF = []
#    print(":",np.asarray(A))
#    print(":",np.asarray(Ranking))
#    print(":",doc_length)
#    print(":",unique_terms)
    for i in range(len(A)):
        RTF.append(0 if math.isnan( (A[i]/doc_length) * math.log(Ranking[0][i] * (unique_terms / A[i]))) else (A[i]/doc_length) * math.log(Ranking[0][i] * (unique_terms / A[i])))
    return RTF


def TFRTF(tf, rtf):
    """
    """
    tfrtf = []
    for i in range(len(tf)):
        rtfrtfdoc = []
        for j in range(len(tf[i])):
            rtfrtfdoc.append(tf[i][j] * rtf[i][j])
        tfrtf.append(rtfrtfdoc)

    return tfrtf

def rankfun(A):
    # Number of times the term occurs in the document
    rank = []
#    for i in range(len(A)):
#        rank.append(ss.rankdata(A[i], method="dense"))
    rank.append(ss.rankdata(A, method="dense"))
    # replacing one value
    
    rank = np.asarray(rank)
#    print(rank)
#    ranknormaliz = []
#    for i in range(len(rank)):
#                
##        for i in range(len(l)):
#        if rank[i] == 1:
#            rank[i] = 0
#        else:
#            rank[i] -= 1
#        ranknormaliz.append(rank)
    # rank[A == 1] = 0
    # print(ranknormaliz)
    # print(type(rank))
    return rank


def maxtotalfreqforgraoup_fun(c_partitioned_matrix):
    return max([sum(i) for i in c_partitioned_matrix])
#
#def largestsizeoftermgroup_fun(c_partitioned_matrix):
#    return c_partitioned_matrix.index(max([sum(i) for i in c_partitioned_matrix]))

def NewTermFreq_Fun(term_freq,c_partitioned_matrix):
    new_TF = [] 
    term_freq = np.asarray(term_freq)
    maxtotalfreqforgraoup = maxtotalfreqforgraoup_fun(c_partitioned_matrix)
#    thelargestgroup = largestsizeoftermgroup_fun(c_partitioned_matrix)
#    print(thelargestgroup,"=====")
    for i in range(len(term_freq)):
        largestingroup = max((l[i]) for l in c_partitioned_matrix)
#        print('largestingroup = ', largestingroup)
        new_TF.append(term_freq[i] + .5 * (maxtotalfreqforgraoup - largestingroup) )
    return new_TF
    
def main(path, name, version,model):
#    path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/'
    path = path
#    path = r'F:\Arabic dataset prepared\ArabicAC\version5.1\short/'
#    name = 'ArabicACc'
#    name ='ArabicCorpusACTwosentence1cleaned'
    name = name
    with open(path+name+'.txt', 'r',
              encoding="utf-8-sig") as f:
        raw_docs = [line.replace("\n", "") for line in f]

    text = ''
    #    print(raw_docs[0:2])

    vec = CountVectorizer()
    X = vec.fit_transform(raw_docs)
    #    print(X)
    #    print(vec.vocabulary_)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    #    print(df["neutr"][:])
    #        text = text + "\n"
    #        file.write(text)
    #    print(df.shape)

    #    print(df.head())
    #    getting the title of the datframe
    # vocabularies = df.columns
    # bmx.writingVocabulary(vocabularies,
    #                   r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/ArabicCorpusACTwosentence1tfrfstvocab.txt')
    # print(vocabularies)
    numpy_matrix = df.as_matrix()
    #    print(numpy_matrix)
    #    print(numpy_matrix.shape)
    tf = termfrequency(numpy_matrix)
    
    for index, row in df.iterrows():
        doc= []
#        print(index)
        doc_freq = []
        for i in range(len(df.columns)): 
            if row[i] != 0:
                doc.append(row.index[i])
                doc_freq.append(row.values[i])
#                print(row.index[i])
#                the term in the document
#        print(doc)
#                the semantic scores for all pairs-term in the document
        semantic_score_matrix = ssc.semantic_scores.semantic_score_matrix(doc,version,model)
#        clusteriing the term in the documents usnig  fuzzy c means clustering
        c_partitioned_matrix  = func.clustering(semantic_score_matrix)
        
#        print(c_partitioned_matrix)
#        the lenghth of the document
        doc_length = sum(np.array(row.values))
#        print(doc_length)
#        print("row.values : ",row.values)
        tfrow = np.array( tf[index])
        doc_freq = tfrow[np.nonzero(tfrow)] 
#        the number of the unique term in the document
        unique_terms = len(doc_freq)
#        calculating the new term frequency
        newTermfreq = NewTermFreq_Fun(doc_freq,c_partitioned_matrix)
#        print(newTermfreq)
#        print("===============================================================")
#        print(index, row)
#        doc.join((df == 38.15).idxmax(axis=1)[0])
    # print(numpy_matrix)
        ranking = rankfun(newTermfreq)
        ranking = np.asarray(ranking)
    # weighted_array1 = weighted_matrix2.toarray()
    # weighted_matrix5 = weights.log_WLDA(numpy_matrix)
        newTermfreq = np.asarray(newTermfreq)
#        print(newTermfreq)
#        print(ranking)
        doc_tfrstf = TFRSTF(newTermfreq, ranking, doc_length, unique_terms)
#        print("doc_tfrstf : ",doc_tfrstf[1])
        for i in range(unique_terms):
            df.loc[index,doc[i]] = doc_tfrstf[i]
#            print(df.at[index,doc[i]])
    tfrstf = df.as_matrix()
#    print(df.head)
#    print(numpy_matrix)
#    del df
    total_tfrstf=np.asarray(numpy_matrix).sum(axis=0)
    
#    print(total_tfrstf)
    
#    total_tfrtf = np.asarray(tfrtf).sum(axis=0)
#    # print(total_tfrtf)
#    del numpy_matrix
    bmx.writingTotalweighttofile(total_tfrstf,path+'tfrfst.txt')
    bmx.writingtofile(tfrstf,path+'singletfrfst.txt')

    #    print("weighted_matrix5 : ",weighted_matrix5)
    # weighted_matrix5 = normalize(weighted_matrix5)
    #    print("weighted_matrix5 after normalize : ", weighted_matrix5)







