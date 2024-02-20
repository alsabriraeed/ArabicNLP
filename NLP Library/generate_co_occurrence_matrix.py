# -*- coding: utf-8 -*-
# -*- coding: utf-8-sig -*-
"""
Created on Thu Aug  1 13:31:33 2019

@author: Raeed
"""
#import re
#text = 'z 23rwqw a 34qf34 hت 343 fsdfd'
#
#print(' '.join( [w for w in text.split() if len(w)>2] ))
##print(re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', text))

import numpy as np
import nltk
from nltk import bigrams
import itertools
import pandas as pd
from nltk.tokenize import word_tokenize
 
def generate_co_occurrence_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}
 
    # Create bigrams from all words in corpus
    bi_grams = list(bigrams(corpus))
 
    # Frequency distribution of bigrams ((word1, word2), num_occurrences)
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
 
    # Initialise co-occurrence matrix
    # co_occurrence_matrix[current][previous]
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))
 
    # Loop through the bigrams taking the current and previous word,
    # and the number of occurrences of the bigram.
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_index[current]
        pos_previous = vocab_index[previous]
        co_occurrence_matrix[pos_current][pos_previous] = count
    co_occurrence_matrix = np.matrix(co_occurrence_matrix)
 
    # return the matrix and the index
    return co_occurrence_matrix, vocab_index
 
with open(r'C:\Users\Raeed\workspace\STTM\STTM\dataset\Tweetpractice.txt', 'r',encoding="utf8") as f:
    raw_docs = [word_tokenize(line.replace("\n","").strip()) for  line in f] 
#raw_docs =list(word_tokenize(' '.join(raw_docs)))
print(raw_docs)
#text_data = [['Where', 'Python', 'is', 'used'],
#             ['What', 'is', 'Python' 'used', 'in'],
#             ['Why', 'Python', 'is', 'best'],
#             ['What', 'companies', 'use', 'Python']]
text_data = raw_docs
 
# Create one list using many lists
data = list(itertools.chain.from_iterable(text_data))
matrix, vocab_index = generate_co_occurrence_matrix(data)
 
data_matrix = pd.DataFrame(matrix, index=vocab_index,
                             columns=vocab_index)
data_matrix =data_matrix.sort_index()
print(data_matrix)