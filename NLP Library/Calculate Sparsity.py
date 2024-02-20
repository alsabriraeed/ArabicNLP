# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:52:23 2019

@author: Raeed

Sparsicity is nothing but the percentage of non-zero datapoints in the document-word matrix, that is data_vectorized.

Since most cells in this matrix will be zero, I am interested in knowing what percentage of cells contain non-zero values.
"""


import weights
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import wordTowordcoocurrence
from math import ceil
from TF_RTF import termfrequency
from scipy.spatial import distance
import OkapiBM25
import TF_RTF





def writingTotalweighttofile(weighted_array, filepath):
    #    print(len(weighted_array))
    filefortotalweight = open(filepath, "w", encoding="utf8")
    stringweight = ""
    for n in range(len(weighted_array)):
        stringweight += str(ceil(weighted_array[n] * 1000000) / 1000000.0) + " "

    filefortotalweight.write(stringweight + "\n")
    filefortotalweight.close()


if __name__ == '__main__':
       # datasets =['ArabicAC','AlKhaleej','AlArabiya','Akhbarona']
    # versions =["version2","version2.1","version3","version3.1","version4",
    # 				"version4.1","version5","version5.1","version6","version6.1","version7","version7.1"]

    datasets = ['ArabicAC']
    versions = ['version3','version3.1']
    #    versions =["version3","version3.1","version4",
    #				"version4.1","version5","version5.1","version6","version6.1","version7","version7.1"]


    datasettype = 'short'  # 'long'

    for dataset in datasets:
        for version in versions:

            #            path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/'
            #    path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/'
            path = 'F:\Arabic dataset prepared\\' + dataset + r'\\' + version + '\\' + datasettype + '\/'
            #            name = 'forpractice'
            #    name = 'ArabicCorpusACTwosentence1cleaned'
            name = dataset + 'c'
            #            name =dataset
            with open(path + name + '.txt', 'r', encoding="utf-8-sig") as f:
                raw_docs = [line.replace("\n", "") for line in f]

            vec = CountVectorizer()
            X = vec.fit_transform(raw_docs)
            # Materialize the sparse data
            data_dense = X.todense()

            # Compute Sparsicity = Percentage of Non-Zero cells
            print("Sparsicity: ", version ,"  ",((data_dense > 0).sum() / data_dense.size) * 100, "%")

#             number of the unique words
            df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
            #    del X

            vocabularies = df.columns
            print("Number of the unique Words: ", version, "  ",len(vocabularies))






