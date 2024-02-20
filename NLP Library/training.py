# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:35:29 2019

@author: Raeed
"""
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

with open(r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/forpractice.txt', 'r', encoding="utf-8-sig") as f:
    raw_docs = [line.replace("\n", "") for line in f]
data = []

for doc in raw_docs:
    temp = []
    for j in word_tokenize(doc):
        temp.append(j)
    data.append(temp)

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count=1, size=100,
                                window=10, sg=1)

model2.save(
    r'C:\Users\Raeed\Desktop\Web-Crawling-with-Clustering-FuzzyCMeans-and-SilhouetteCoefficient-master\Web-Crawling-with-Clustering-FuzzyCMeans-and-SilhouetteCoefficient-master/mymodel1')
new_model = gensim.models.Word2Vec.load(
    r'C:\Users\Raeed\Desktop\Web-Crawling-with-Clustering-FuzzyCMeans-and-SilhouetteCoefficient-master\Web-Crawling-with-Clustering-FuzzyCMeans-and-SilhouetteCoefficient-master/mymodel1')

with open(r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/forpractice.txt', 'r', encoding="utf-8-sig") as f:
    raw_doc = [line.replace("\n", "") for line in f]
docs = []

for doc in raw_doc:
    temp = []
    for j in word_tokenize(doc):
        temp.append(j)
    docs.append(temp)

# for doc in data:
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


# Print results
# print("Cosine similarity between 'alice' " +
#       "and 'wonderland' - Skip Gram : ",
#       new_model.similarity('عالم', 'ناس'))

# print("Cosine similarity between 'alice' " +
#       "and 'machines' - Skip Gram : ",
#       model2.similarity('alice', 'machines'))
