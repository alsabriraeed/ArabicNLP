# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:27:39 2019

@author: Raeed
#"""


from pylab import *
import numpy as np
import numpy
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse import * 


def create_co_occurences_matrix(allowed_words, documents):
#    print(f"allowed_words:\n{allowed_words}")
#    print(f"documents:\n{documents}")
    word_to_id = dict(zip(allowed_words, range(len(allowed_words))))
    documents_as_ids = [np.sort([word_to_id[w] for w in doc if w in word_to_id]).astype('uint32') for doc in documents]
    row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in doc] for i, doc in enumerate(documents_as_ids)]))
    data = np.ones(len(row_ind), dtype='uint32')  # use unsigned int for better memory utilization
    max_word_id = max(itertools.chain(*documents_as_ids)) + 1
    data = csr_matrix((data, (row_ind, col_ind)), shape=(len(documents_as_ids), max_word_id))  # efficient arithmetic operations with CSR * CSR
    del row_ind
    del col_ind
    del documents_as_ids
    data = data.T * data  # multiplying docs_words_matrix with its transpose matrix would generate the co-occurences matrix
    data.setdiag(0)
#    print(f"words_cooc_matrix:\n{words_cooc_matrix.todense()}")
    return data, word_to_id 

def calprocoocurenofwords(wTowcoocurence):
#    print("wTowcoocurence: ",type(wTowcoocurence))
#    df = pd.DataFrame(wTowcoocurence)
#    return ((df.T / df.T.sum()).T)
    return wTowcoocurence*1./np.sum(wTowcoocurence, axis=0)

#def caculateentropy(procoocurenofwords):
##    print(type(procoocurenofwords))
#    print(procoocurenofwords)
#    procoocurenofwords1 = procoocurenofwords.iloc[:,0]
#    for row in range(procoocurenofwords.shape[1]):
#        sumforeachword = 0.0
#        count = 0
#        for col in range(procoocurenofwords.shape[1]):
#            if procoocurenofwords.iat[row,col]== 0:
#                continue
#            count+=1
#            sumforeachword += procoocurenofwords.iat[row,col] * log( procoocurenofwords.iat[row,col])
##        print(sumforeachword)
#        normalizedentropy = -1 * ((-1 * sumforeachword)/abs(log(count)))
##        print(normalizedentropy)
#        procoocurenofwords1.iloc[row] = (-1 * round_(normalizedentropy,decimals = 9))
#    return procoocurenofwords1

def caculateentropy(procoocurenofwords):
#    print(type(procoocurenofwords))
#    print(procoocurenofwords)
    counteachnonzerocolumns = procoocurenofwords.astype(bool).sum(axis=1)
    procoocurenofwords = procoocurenofwords.replace(0.0,1.0)
#    print(procoocurenofwords)
#    print(counteachnonzerocolumns)
    sumforeachword =[]
    for row in range(procoocurenofwords.shape[1]):
        sumforeachword.append((procoocurenofwords.loc[row] * np.log(procoocurenofwords.loc[row])).sum(axis=0))
#        print((procoocurenofwords[row] * log(procoocurenofwords[row])).sum())
#    counteachnonzerocolumns = procoocurenofwords.astype(bool).sum(axis=1)
#    print(np.asarray(sumforeachword))
#    print(np.asarray(counteachnonzerocolumns))
    del procoocurenofwords
    sumforeachword = -1 * ((-1 * np.asarray(sumforeachword))/abs(np.log(np.asarray(counteachnonzerocolumns))))
#    print(normalizedentropy)
    
    sumforeachword[~np.isfinite(sumforeachword)] = 0

#    normalizedentropy[numpy.isneginf(normalizedentropy)] = 0
#    normalizedentropy[normalizedentropy == -inf] = 0
    sumforeachword = (-1 * np.round_(sumforeachword,decimals = 9))
#    normalizedentropy[normalizedentropy == inf] = 0
#    print(normalizedentropy)
    return sumforeachword

