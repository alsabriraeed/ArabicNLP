#import scipy
#from pylab import *
#from scipy.sparse import * #imports all the sparse matrix types and their functions
#import numpy as np
#import math
#from scipy.spatial import distance
#import math
#print(log2(7/3))
#from math import ceil
#
#num = 0.1111111111000
#num = ceil(num * 1000000000) / 1000000000.0
#print(num)
#import math
#import script as s
# x = (1/9645) / 1.0368066355624676E-4 * math.log( (1/9645) / 1.0368066355624676E-4)
#
# print(x)
#
# def f(text):
#     text = text.replace(s.ar_ZERO, '').replace(s.ar_ONE, '').replace(s.ar_TWO,'').replace(s.ar_THREE,'').replace(s.ar_FOUR,'').replace(s.ar_FIVE,'').replace(s.ar_SIX,'').replace(s.ar_SEVEN,'').replace(s.ar_EIGHT,'').replace(s.ar_NINE,'')
#     return text
#
# print(f("&jhjgg:٢٠١٣"))
#
## print( model.similarity('hot', 'cold'))
#from nltk.corpus import wordnet
#
#list1 = ['like']
#list2 = ['hope']
#list = []
#
#for word1 in list1:
#    for word2 in list2:
#        wordFromList1 = wordnet.synsets(word1)
#        wordFromList2 = wordnet.synsets(word2)
#        if wordFromList1 and wordFromList2:
#            s = wordFromList1[0].wup_similarity(wordFromList2[0])
#            list.append(s)
#
#print(list)
#
#str1 = '‏ ليسجل‏‏ جنيه لشراء و‏‏ جنيه لبيي'
#print(' '.join( [w.replace(" ", "").replace("  ", "") for w in str1.split() if len(w.replace(" ", ""))> 2 ] ))

#print(len('ل'))
#
#value= [1,2,3,4,5]
#dollarizer = [float(value[1 : -1]) for va in value]
#print(value)

#import numpy as np
#
#a = np.array([[ 5, 1 ,3], 
#                  [ 1, 1 ,1], 
#                  [ 1, 2 ,1]])
#b = np.array([1, 2, 3])
#print(a)
#print(b)
#
#print (a.dot(b))

#from sklearn import metrics
#labels_true = [0, 0, 0, 1, 1, 1]
#labels_pred = [0, 0, 0, 1, 1, 1]
#
#print(metrics.homogeneity_score(labels_true, labels_pred))


#x= ['Wikipedia', 'file', 'in', ':', 'F', ':', '\\Arabic', 'dataset', 'prepared\\wiki\\version2\\wiki.txt', 
#    'Results', 'for', ':', 'F', ':', '\\Arabic', 'dataset', 'prepared\\ArabicAC\\version2\\short\\tfrtfresults\\ArabicACctfrtf1.topWords',
#    'Coherence', ':', '0.09995623735161811', '--', '-',
#    'Mean', 'Coherence', ':', '0.09995623735161811', ',', 'standard', 'deviation', ':', '0.0']

#x= ['Golden-labels', 'in', ':', 'F', ':', '\\Arabic', 'dataset', 'prepared\\ArabicAC\\version2\\short\\ArabicAC_label.txt',
# 'Results', 'for', ':', 'F', ':', '\\Arabic', 'dataset', 'prepared\\ArabicAC\\version2\\short\\tfrtfresults\\ArabicACctfrtf1.theta',
# 'Accuracy', ':', '50.925925925925924', '--', '-', 
# 'Mean', 'accuracy', ':', '50.925925925925924', ',', 'standard', 'deviation', ':', '0.0']
##x = ['Golden-labels', 'in', ':', 'F', ':', '\\Arabic',
##     'dataset', 'prepared\\ArabicAC\\version2\\short\\ArabicAC_label.txt', 'Results', 'for', ':', 'F', ':',
##     '\\Arabic', 'dataset', 'prepared\\ArabicAC\\version2\\short\\pmiokapibm25BDCresults\\ArabicACcpmiokapibm25BDC1.theta',
##     'Purity', ':', '0.6611111111111111', 'NMI', ':', '0.4833029501358129', '--', '-', 'Mean', 'purity', ':', 
##     '0.6611111111111111', ',', 'standard', 'deviation', ':', '0.0', 'Mean', 'NMI', ':', 
##     '0.4833029501358129', ',', 'standard', 'deviation', ':', '0.0']
#print(len(x))
##print( x[27],'|', x[32],'|', x[36],'|',x[41])
#print( x[24],'|',x[29])
#print( x[25],'|',x[30])
#import scipy
#from pylab import *
#from scipy.sparse import * #imports all the sparse matrix types and their functions
#import numpy as np
#import math
#print(log2(4/1))
#x = np.random.rand(2, 2)
#x=scipy.sparse.csc_matrix(x,dtype=float64)
#y = np.asarray([55,5,6])
#
#x[:,1] *=y[0]
#print(x)
#import numpy as np
#m = np.array([[1,2,3],[4,5,6],[7,8,9]])
#c = np.array([0,1,2])
#print(m * c)
#print(m * c[:, np.newaxis])

#import numpy as np
#X = np.array([[1, 0, 0],
#              [0, 2, 2],
#              [3, 0, 0]])
#
#Y=  np.array([1, 0, 0,4,5,0,6])
### " ".join(str(x) for x in np.asarray(np.nonzero(Y))[0,:])
#print(" ".join(str(x) for x in Y[np.nonzero(Y)]))
#print(" ".join(str(x) for x in np.asarray(np.nonzero(Y))[0,:]))


#allowed_words = ['A', 'B', 'C', 'D']
#documents = [['A', 'B'], ['C', 'B', 'K'],['A', 'B', 'C', 'D', 'Z']]
#allowed_words =['china',  'hello',   'hi',  'siad',  'tell' , 'waw' , 'yemen']
##print(wordTowordcoocurrence.calculatewordtowordcoocurrence1())
#documents = [['hi', 'hello', 'hi', 'tell'],['siad', 'yemen', 'china', 'hi'], ['waw', 'hello', 'hi']]
#words_cooc_matrix, word_to_id = create_co_occurences_matrix(allowed_words, documents)
#
#print(words_cooc_matrix.toarray())
#import wordTowordcoocurrence
#print(wordTowordcoocurrence.calculatewordtowordcoocurrence1(['hi hello hi tell', 'siad yemen china hi', 'waw hello', 'hi']))
# import pandas as pd
# import numpy as np
# from pylab import *
# #Create a DataFrame
# #df1 = {[2,3],
# #       [2,0]}
# #
# #df1 = pd.DataFrame(df1)
# #for row in range(df1.shape[1]):
# #    print(((df1[str(row)] * log(df1[str(row)])).sum()))
# #print(df1.astype(bool).sum(axis=1))
# #print(log(1))
#
# #print(((df1.values * log(df1.values)).sum()))
#
# #print(df1)
# #print(0.125*log(0.125)+0.333333*log(0.333333)+0.166667*log(0.166667))
# #print(0.333333*log(0.333333)+0.333333*log(0.333333)+0.333333*log(0.333333))
# ##print(df1)
# #print(2 *log(sqrt(7)/4))
# #print(log(sqrt(10000)/4))
# #import numpy as np
# #
# #v = [[1,2,3],[3,4,5],[6,7,1]]
# #vv = np.array(v)
# #
# #print(vv*5)
#
# #
# #mat = np.array([[1, 0], [3, 4]])
# #print(type(mat))
# #cons = 13
# #mat= mat- cons
# #print(mat)
# #x1 = np.arange(9.0).reshape((3, 3))
# #print(x1)
# #
# #x2 = np.arange(3.0)
# #print(x2)
# #print(np.subtract(x1, x2))
# import numpy as np
# import math
# x = np.array([[0.0, 0.0], [1.0, 0.0]])
# norms = np.array([np.linalg.norm(a) for a in x])
# print(norms)
# nonzero = norms > 0
# nonzero1 = x > 0
# print(nonzero1)
# print(nonzero)
# x[nonzero] /= 1
# print(x)
#
# print(0- -1)

print(round((1*77+2*90+3*94+3*94+2*95+2*96+97*3+2*97+3*92+2*95+2*94.5+3*95+2*95++2*97+2*100)/(34),1))
print(0/1)