# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:33:03 2019

@author: Raeed
"""
import scipy
from pylab import *
from scipy.sparse import * #imports all the sparse matrix types and their functions
import numpy as np
import math
from scipy.spatial import distance
import math
print(log2(2.25))
from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

document1 = tb("""hi hello hi tell""")

document2 = tb("""siad yemen china hi""")

document3 = tb("""waw hello""")
document4 = tb("""hi lol deed""")

bloblist = [document1, document2, document3, document4]
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#    for word, score in sorted_words[:3]:
#        print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))
    print(sorted_words)