# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:37:31 2019

@author: raeed
"""

import weights
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import wordTowordcoocurrence
from math import ceil

def writingtofile(weighted_array,path):
#    print(len(weighted_array[0]))
    fileforclass = open(path,"w",encoding="utf8")
    for n in range(len(weighted_array)):
        stringweight =""
        for j in range(len(weighted_array[n])):
           stringweight += str(weighted_array[n][j]) + " "
 
        fileforclass.write(stringweight +"\n")
    fileforclass.close()
    
def writingTotalweighttofile(weighted_array,filepath):
#    print(len(weighted_array))
    filefortotalweight = open(filepath,"w",encoding="utf8")
    stringweight =""
    for n in range(len(weighted_array)):
        stringweight += str( ceil(weighted_array[n] * 1000000) / 1000000.0 ) + " "
 
    filefortotalweight.write(stringweight +"\n")
    filefortotalweight.close()
    
def writingTotalweighttofile_adding(weighted_array,weighted_array1):
    print(len(weighted_array))
    print(len(weighted_array1))
    filefortotalweight = open('F:\output/ArabicCorpusAClogentropytfidf.txt',"w",encoding="utf-8-sig")
    stringweight =""
    for n in range(len(weighted_array)):
        stringweight += str(weighted_array[n] + weighted_array1[n]) + " "
 
    filefortotalweight.write(stringweight +"\n")
    filefortotalweight.close()
  
def writingTotalweighte_adding(weighted_array, weighted_array1, weighted_array2 ):
#    print(len(weighted_array))
    filefortotalweight = open('F:\output/ArabicCorpusACWeightlogentropythree.txt',"w",encoding="utf8")
    stringweight =""
    for n in range(len(weighted_array)):
        stringweight += str(weighted_array[n] + weighted_array1[n]+weighted_array2[n]) + " "
 
    filefortotalweight.write(stringweight +"\n")
    filefortotalweight.close()

def writingVocabulary(vocab_array,path ):
#    print(len(vocab_array))
    filevocab = open(path ,"w",encoding="utf8")
    stringweight =""
    for n in range(len(vocab_array)):
        stringweight += str(vocab_array[n] ) + " "
 
    filevocab.write(stringweight)
    filevocab.close()
    
    
def writingVocabularybyword(vocab,vocab2,path ):
    filevocab = open(path ,"w",encoding="utf8")
    for n in range(len(vocab)):
        filevocab.write(vocab[n] + ":    :"+ vocab2[n]+ "\n")
 
    filevocab.close()
    

def normalize(weighted_matrix5):
    max_value = max(weighted_matrix5)
    min_value = min(weighted_matrix5)
    
    for i in range(len(weighted_matrix5)):
        weighted_matrix5[i] = (weighted_matrix5[i] - min_value)/ (max_value -  min_value)
    
    return weighted_matrix5


def normalize_singleweights(weighted_matrix5):
    
    normweighted_matrix = []
    for i in range(len(weighted_matrix5)):
        weightedarray = np.array( weighted_matrix5[i])
        max_value = max(weightedarray[np.nonzero(weightedarray)])
        min_value = min(weightedarray[np.nonzero(weightedarray)])
#        print("max_value", max_value)
#        print("min_value", min_value)
        normweighted_matrix_row =[]
        for j in range(len(weighted_matrix5[i])):
            if weighted_matrix5[i][j] == 0 :
                normweighted_matrix_row.append(0)
            else:
                if max_value == min_value:
                    normweighted_matrix_row.append(0 + .1)
                else:
                    normweighted_matrix_row.append(.1 + ((weighted_matrix5[i][j] - min_value)/ (max_value -  min_value)))
        normweighted_matrix.append(normweighted_matrix_row)     
    return normweighted_matrix 

def calculateLogwlda_entropy(logwldaweight, entropyweight):
    A=np.zeros(entropyweight.shape)
    print(entropyweight.shape)
    for i in range(entropyweight.shape[0]):
        for j in range(entropyweight.shape[1]):
            if entropyweight[i][j]== 0. :
                continue
            else:
                A[i][j] = entropyweight[i][j] + logwldaweight[j]
    return A
    

if __name__ == '__main__':
    with open(r'F:\output\AlArabiya\AlArabiyalightstemmer/AlArabiyaar_Lightstemercleanshort20.txt', 'r',encoding="utf-8-sig") as f:
        raw_docs = [line.replace("\n","") for  line in f]
    
#    text =''
#    print(raw_docs[0:2])
    
    vec = CountVectorizer()     
    X = vec.fit_transform(raw_docs)
#    print(X)
#    print(vec.vocabulary_)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    del X
#    print(df["neutr"][:])
#        text = text + "\n"
#        file.write(text)
#    print(df.shape)

#    print(df.head())
#    getting the title of the datframe
    vocabularies = df.columns
    writingVocabulary(vocabularies,r'F:\output\AlArabiya\AlArabiyalightstemmer/AlArabiyaar_Lightstemercleanshort20cewvocab.txt')
#    print(vocabularies[0])
    numpy_matrix = df.as_matrix()
#    print(numpy_matrix)
#    print(numpy_matrix.shape)
    del df 
#    weighted_matrix = weights.pmi(numpy_matrix)
#    weighted_matrix1 = weights.logentropy(numpy_matrix)
#     weighted_matrix2 = weights.tfidf(numpy_matrix)
#    weighted_matrix3 = weights.logtf(numpy_matrix)
#    weighted_matrix4 = weights.logentropy2(numpy_matrix)

    weighted_matrix5 = weights.log_WLDA(numpy_matrix)
    del numpy_matrix
#    print("weighted_matrix5 : ",weighted_matrix5)
    weighted_matrix5 = normalize(weighted_matrix5)
    writingTotalweighttofile(weighted_matrix5,r'F:\output\AlArabiya\AlArabiyalightstemmer/AlArabiyaar_Lightstemercleanshort20log_WLDA.txt')
   # del weighted_matrix5
#    print("weighted_matrix5 after normalize : ", weighted_matrix5)
#    print(df.columns)
    weighted_matrix6 = wordTowordcoocurrence.main(raw_docs)
    del vocabularies
    del raw_docs
    weighted_matrix6 = normalize(weighted_matrix6)
#    print("weighted_matrix6 after normalize : ",list( weighted_matrix6.values))
#    weighted_matrix6 = weighted_matrix6.drop(weighted_matrix6.index[[len(weighted_matrix6) - 1,len(weighted_matrix6) - 2]])
    #weighted_matrix6 = weighted_matrix6.drop(weighted_matrix6.index[len(weighted_matrix6) - 1])
    #vocabularies1 = weighted_matrix6.index
    
#    del vocabularies1[19230]
#    for i in range(len(vocabularies)):
#        if vocabularies1[i] != vocabularies[i]:
             #writingVocabularybyword(vocabularies[i],vocabularies1[i])
    # writingVocabularybyword(vocabularies,vocabularies1,r'F:\output\AlArabiya/AlArabiyashortkhojacleaned20ccompvocab.txt')
#            print(vocabularies1[i])  
#            break
#            count+=1
    #filevocab.close()

#    print(count)
#    print(len(vocabularies))
    #print(len(vocabularies1))
#    print("len :", len(weighted_matrix6))
#    print(weighted_matrix6)
#    weighted_matrix6 =weights.checktype(weighted_matrix6)
    logldaentropyweight = np.multiply(weighted_matrix5,weighted_matrix6)
    print(logldaentropyweight.shape)
    #logldaentropyweight = list(logldaentropyweight)
#    print(list(logldaentropyweight.values))
#    weighted_matrixpractice1 = weights.compute_weight(X)
#    print(weighted_matrixpractice1)
#    print(weighted_matrixpractice)
#    weighted_array = weighted_matrix.toarray()
#    weighted_array1 = weighted_matrix1.toarray()
#     weighted_array2 = weighted_matrix2.toarray()
#    weighted_array3 = weighted_matrix3.toarray()
#    weighted_array4 = weighted_matrix4.toarray()
#    print(weighted_matrix.shape)
#    logwldaentropyweight= calculateLogwlda_entropy(weighted_matrix5,weighted_array2)
#    print(weighted_matrix5)
#    print(weighted_array1)
#    print(logwldaentropyweight)
#    tfentropyweight = np.add(weighted_matrixpractice1,weighted_array2)
#    tfentropyweight = np.add(tfentropyweight,weighted_array2)
#    tfentropyweight =np.add(tfentropyweight, weighted_array1)
#    print(weighted_array)
#    
#    print(weighted_array2)
#    print(weighted_array3)
#    print(weighted_array4)
#    print(weighted_matrixpractice1)
#    print(tfentropyweight)
#    to ensure that the weight for each word is correct
#    total_weight=np.asarray(weighted_array).sum(axis=0)
#    total_weight1=np.asarray(weighted_array1).sum(axis=0)
#     total_weight2=np.asarray(weighted_array2).sum(axis=0)
#    total_weight3=np.asarray(weighted_array3).sum(axis=0)
#    total_weight4=np.asarray(weighted_array4).sum(axis=0)
#    del weighted_array
#    print(total_weight.shape)
#    print(total_weight[7341])
#    print(total_weight[7343])
#    print(total_weight[5097])
#    print(sum(weighted_matrix["5095"]))
    
#    total_weightpractice=np.asarray(weighted_arraypractice).sum(axis=0)
    
#    writingtofile(logldaentropyweight)
    writingTotalweighttofile(logldaentropyweight,r'F:\output\AlArabiya\AlArabiyalightstemmer/AlArabiyaar_Lightstemercleanshort20CEW.txt')
    #del logldaentropyweight
#    print()
#    print(total_weightpractice)
#    writingTotalweighttofile(total_weightpractice)
    
#    writingTotalweighttofile_adding(total_weight1,total_weight2)
#    writingTotalweighte_adding(total_weight,total_weight1, total_weight2)
    
    
    
    # writingtofile( weighted_array2,r'F:\output\AlArabiya/AlArabiyashortkhojacleaned20csingletfidf.txt')

#    print(type(weighted_array))
#    print((weighted_array.shape))
#    print(weighted_array[0][0])






