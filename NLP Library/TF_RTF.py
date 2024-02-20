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


def RTF(A, Ranking):
    """
    """
    lendoc = np.count_nonzero(A, axis=1)
    # Number of times the term occurs in the document
    RTF = []
    for i in range(len(A)):
        rankdoc = []
        for j in range(len(A[i])):
            if A[i][j] == 0:
                rankdoc.append(0)
            else:

                rankdoc.append(0 if math.isnan(math.log(Ranking[i][j] * (lendoc[i] / A[i][j]))) else math.log(Ranking[i][j] * (lendoc[i] / A[i][j])))
        RTF.append(rankdoc)

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
    for i in range(len(A)):
        rank.append(ss.rankdata(A[i], method="dense"))
    # replacing one value
    rank = np.asarray(rank)
    ranknormaliz = []
    for l in rank:
        for i in range(len(l)):
            if l[i] == 1:
                l[i] = 0
            else:
                l[i] -= 1
        ranknormaliz.append(l)
    # rank[A == 1] = 0
    # print(ranknormaliz)
    # print(type(rank))
    return ranknormaliz


def main(path,name):
    path = path
#    r'F:\Arabic dataset prepared\ArabicAC\version7.1\short/'
    name = name
#    'ArabicACc'
    with open(path + name + '.txt', 'r',
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
    vocabularies = df.columns
    # bmx.writingVocabulary(vocabularies,
    #                   r'F:\output\ArabicCorpusACT/ArabicCorpusACcleanedtfrtfvocab.txt')
    # print(vocabularies)
    numpy_matrix = df.as_matrix()
    #    print(numpy_matrix)
    #    print(numpy_matrix.shape)
    del df
    # print(numpy_matrix)
    tf = termfrequency(numpy_matrix)
    del numpy_matrix
    ranking = rankfun(tf)
    ranking = np.asarray(ranking)
    # weighted_array1 = weighted_matrix2.toarray()
    # weighted_matrix5 = weights.log_WLDA(numpy_matrix)
    tf = np.asarray(tf)
    rtf = RTF(tf, ranking)
    # print(tf)
    # print(rtf)
    tfrtf = TFRTF(tf, rtf)
    # print(tfrtf)
#    calling the function to normalize the weight for each term in the document
#    tfrtf = bmx.normalize_singleweights(tfrtf)
#    print(tfrtf)
    total_tfrtf = np.asarray(tfrtf).sum(axis=0)
#    here we normalize the weighting term for all terms
#    total_tfrtf = bmx.normalize(total_tfrtf)
    bmx.writingtofile(tfrtf, path + 'singletfrtf.txt')
    # print(total_tfrtf)
    del tfrtf
#    del numpy_matrix
    bmx.writingTotalweighttofile(total_tfrtf,path + 'tfrtf.txt')

    #    print("weighted_matrix5 : ",weighted_matrix5)
    # weighted_matrix5 = normalize(weighted_matrix5)
    #    print("weighted_matrix5 after normalize : ", weighted_matrix5)







