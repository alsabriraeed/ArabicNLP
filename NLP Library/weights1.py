"""
weights.py
A collection of weightings to use with term-document matrices.
See documentation of each method for more information

Methods:
    - pmi(A) : Pointwise Mutual Information weighting method (Bader 2010)
    - logentropy(A) : Log-Entropy weighting combining local log term 
        frequency weighting and global entropy weighting (Bader 2010)
    - tf(A) : Term frequency local weighting method
    - binary(A) : Binary local weighting method
    - logtf(A) : Log term frequency local weighting method
    - idf(A) : Inverse Document Frequency global weighting method
    - entropy(A) : Entropy global weighting
"""
import scipy
import scipy.sparse
from pylab import *
from scipy.sparse import * #imports all the sparse matrix types and their functions
import numpy as np
import math
#import CPython 
from scipy.spatial import distance
#import numba.cuda
#from  numba  import cuda
def checktype(A):
    """
    Function to check the type of values within the matrix A.  If the values are not 
    type float64, they are converted.  Implemented to prevent calculations to performed 
    on data with types such as int, which would create incorrect results due to rounding.
    
    Arguments:
        A: term by document matrix
        
    Returns:
        Matrix equivalent to A but with all entries of type float64
    """
    if A.dtype.type is not np.float64:
        A=scipy.sparse.csc_matrix(A,dtype=float64)
    return A

def pmi(A):
    """
    Function to convert a term by document matrix A to a weighted term-document matrix E using 
    Pointwise Mutual Information (PMI).  As described by Brett Bader, weights each entry Eij 
    as log2[p(i,j)/(p(i)*p(j))].
    
    Arguments:
        A: term by document matrix
    
    Returns:
        E: Pmi weighted matrix
    """
    A=checktype(A)
    
    dims=A.shape
    E=lil_matrix(dims)
    rowsums=A.sum(1)
    colsums=A.sum(0).T
    total=sum(rowsums)
    row,col=nonzero(A)
    for i in range(len(row)):
        E[row[i],col[i]]=round (float64(log2(A[row[i],col[i]]/(rowsums[row[i]]*colsums[col[i]]/total))), 6)
    return E
         
# L1 - Term frequency (tf); All entries are set to term frequency of term i in document j.
# If ter i appears in document j 4 times and document j has 10 words, then w(i,j)=4/10=.4
#def tf(A):
#    """
#    Local weighting method to apply term frequency (tf) to a term-document matrix A.  Tf(i,j) is 
#    defined as the number of occurences of a term i divided by the total number of words in 
#    document j.
#    
#    Arguments:
#        A: m by n term by document matrix
#    
#    Returns:
#        m by n tf weighted matrix
#    """
#    A=checktype(A)
#    sumtermoccurence = array(1.0/A.sum(0))[0]
##    A = A.toarray()
#    
##    print("A.sum(0)\n",A.sum(0))
##    print("dia_matrix(diag(array(1.0/A.sum(0))[0]))",dia_matrix(diag(array(1.0/A.sum(0))[0])).toarray())
#    
##    print(A.shape)
#    
##    return A*dia_matrix(diag(array(1.0/A.sum(0))[0]))
#    for i in range(A.shape[0]):
#        for j in range(A.shape[1]):
#           
#            if A[i,j] !=0:
#                A[i,j] = A[i,j]* sumtermoccurence[j]
#    return A
#@cuda.jit
def tf(A):
    """
    Local weighting method to apply term frequency (tf) to a term-document matrix A.  Tf(i,j) is 
    defined as the number of occurences of a term i divided by the total number of words in 
    document j.
    
    Arguments:
        A: m by n term by document matrix
    
    Returns:
        m by n tf weighted matrix
    """
#    print(type(A))
    A=checktype(A)
    dims=A.shape
#    sumtermoccurence = array(1.0/A.sum(0))[0]
#    A = A.toarray()
    
#    print("A.sum(0)\n",A.sum(0))
#    print("dia_matrix(diag(array(1.0/A.sum(0))[0]))",dia_matrix(diag(array(1.0/A.sum(0))[0])).toarray())
    
#    print(A.shape)
#    for i in range(A.shape[1]):
#        A[:,i] *=sumtermoccurence[i]
    return A*dia_matrix(diag(array(1.0/A.sum(0))[0]))
#    i,j = cuda.grid(2)
#    n ,m = A.shape
##    out = np.empty_like(A)
#    if 1 <= i < n - 1 and 1 <= j < m - 1:
#         out[i,j] = A[i-1,j-1] * sumtermoccurence[j]
#        
#    for i in range(A.shape[0]):
#        for j in range(A.shape[1]):
          
#            if A[i,j] !=0:
#                A[i,j] = A[i,j]* sumtermoccurence[j]
    return A
    

def updatetf(A):
    """
    Local weighting method to apply term frequency (tf) to a term-document matrix A.  Tf(i,j) is 
    defined as the number of occurences of a term i divided by the total number of words in 
    document j.
    
    Arguments:
        A: m by n term by document matrix
    
    Returns:
        m by n tf weighted matrix
    """
    A=checktype(A)
    dims=A.shape
#    print("A\n",A.toarray())
#    print("A.sum(0)\n",A.sum(0))
#    print("dia_matrix(diag(array(1.0/A.sum(0))[0]))",dia_matrix(diag(array(1.0/A.sum(0))[0])).toarray())
    return A*dia_matrix(diag(array(A.sum(0))[0]))
# L1 - Term frequency (tf); All entries are set to term frequency of term i in document j.
# If ter i appears in document j 4 times and document j has 10 words, then w(i,j)=4/10=.4
def tf1(A):
    """
    Local weighting method to apply term frequency (tf) to a term-document matrix A.  Tf(i,j) is
    defined as the number of occurences of a term i divided by the total number of words in
    document j.

    Arguments:
        A: m by n term by document matrix

    Returns:
        m by n tf weighted matrix
    """
    A = checktype(A)
    dims = A.shape

    return A * dia_matrix(diag(array(1.0 / A.sum(0))[0]))

# L2 - Binary: All entries set to 1    
def binary(data):
    """
    Local weighting method to apply binary weighting to a term-document matrix A.  Binary
    weighting is all nonzero entries are simply set to a value of 1
    
    Arguments:
        data: m by n term by document matrix
    
    Returns:
        A: m by n weighted matrix
    """
    
    data=checktype(data)
    A=data.copy()
    row,col=nonzero(A)
    for i in range(len(row)):
        A[row[i],col[i]]=1.0
    return A

# L3 - Logtf: log(term frequency+1)
def logtf(data):
    """
    Local weighting method to apply log term frequency to a term document matrix A.
    Logtf = log(raw term frequency + 1).  Raw term frequency tf(i,j) is the number 
    of times term i appears in document j.
    
    Arguments:
        data: m by n term by document matrix
    
    Returns:
        A: m by n logtf weighted matrix
    """
    data=checktype(data)
    A=data.copy()
    row,col=nonzero(A)
    for i in range(len(row)):
        A[row[i],col[i]]=log2(A[row[i],col[i]].copy()+1)
    return A

def log_WLDA(data):
    """
    Local weighting method to apply log term frequency to a term document matrix A.
    Logtf = log(raw term frequency + 1).  Raw term frequency tf(i,j) is the number 
    of times term i appears in document j.
    
    Arguments:
        data: m by n term by document matrix
    
    Returns:
        A: m by n logtf weighted matrix
    """
    data=checktype(data)
    A=data.copy()
    #    in this statment is to sum the number of words in the corpus
    sumwordsincorpusrow = A.sum(axis = 1) 
    sumwordsincorpus= sumwordsincorpusrow.sum()
    
    sumwordsincorpuscolumn =np.asarray( A.sum(axis = 0))
#    print(type(sumwordsincorpuscolumn))
#    print(sumwordsincorpuscolumn[0][1])
#    print(sumwordsincorpus)
    row,col=nonzero(A)
#    print(row)
#    print(A.shape[1])
    W=[]
    for i in range(A.shape[1]):
#        A[row[i],col[i]]=log2(A[row[i],col[i]].copy()+1)
        if ((sumwordsincorpuscolumn[0][i]/sumwordsincorpus) > (1/A.shape[1])):
            W.append( - log(sumwordsincorpuscolumn[0][i]/sumwordsincorpus))
        else:
            W.append( - log(1/A.shape[1]))
    return W

def pmi_log_WLDA(data):
 
  
    data=checktype(data)
    A=data.copy()
    #    in this statment is to sum the number of words in the corpus
    sumwordsincorpusrow = A.sum(axis = 1) 
    sumwordsincorpus= sumwordsincorpusrow.sum()
    
    sumwordsincorpuscolumn =np.asarray( A.sum(axis = 0))
#    print(type(sumwordsincorpuscolumn))
#    print(sumwordsincorpuscolumn[0][1])
#    print(sumwordsincorpus)
    row,col=nonzero(A)
#    print(row)
#    print(A.shape[1])
    W=[]
    w1=[]
    for i in range(A.shape[1]):
#        A[row[i],col[i]]=log2(A[row[i],col[i]].copy()+1)
        if ((sumwordsincorpuscolumn[0][i]/sumwordsincorpus) > (1/A.shape[1])):
            W.append( - log(sumwordsincorpuscolumn[0][i]/sumwordsincorpus))
        else:
            W.append( - log(1/A.shape[1]))
    A = A.toarray()
    for i in range(A.shape[0]):
        temp=[]
        for j in range(A.shape[1]):
#            print(A.shape)
#            print(type(A))
#            print(A[i][j])
#            print(type(sumwordsincorpusrow))
#            print(sumwordsincorpusrow.item(i))
#            print(W[i])
            if (A[i][j] !=0):
                temp.append(- log((A[i][j]/sumwordsincorpusrow.item(i))/W[j]))
            else:
                 temp.append(0)
            
        w1.append(temp)
    return w1



# G4 - Inverse document frequency (idf)
def idf(A):
    """
    Inverse document frequency (idf) global weighting method.  idf=log(N/df(i))+1, 
    where N is the number of docuemnts and df(i) is the number of document in which 
    term i occurs.  "term i is now treated as an event that either occurs or does 
    not occur in a given document.  The expression df(i)/N is the probability of that 
    event; and -log(df(i)/N), or equivalently log(N/df(i)), is the negation of the 
    event's probability" - Bader 2010
    
    For use in conjunction with a local weighting method as it returns a vector by
    which each column of the data matrix A must be multiplied.  Specifically used
    best in conjunction with term-frequency (tf), though there is already a defined
    method in this module which does this, tfidf()
    
    Arguments:
        A: m by n term by document matrix
    
    Returns:
        m by 1 idf global weighting vector
    """
    A=checktype(A)
    m,N=A.shape
    dvec=zeros((m,1))
    for i in range(m):
        dvec[i,0]=A[i,:].nnz
#    print("dvec: ", dvec)
#    print("N : ", N)
    try: 
        return log2(N/dvec)
        return (N/dvec)
    except ZeroDivisionError:
        return log2(N/(dvec+1.0))
        return (N/(dvec+1.0))
# G4 - Inverse document frequency (idf)
def updateidf(A):
    """
    Inverse document frequency (idf) global weighting method.  idf=log(N/df(i))+1, 
    where N is the number of docuemnts and df(i) is the number of document in which 
    term i occurs.  "term i is now treated as an event that either occurs or does 
    not occur in a given document.  The expression df(i)/N is the probability of that 
    event; and -log(df(i)/N), or equivalently log(N/df(i)), is the negation of the 
    event's probability" - Bader 2010
    
    For use in conjunction with a local weighting method as it returns a vector by
    which each column of the data matrix A must be multiplied.  Specifically used
    best in conjunction with term-frequency (tf), though there is already a defined
    method in this module which does this, tfidf()
    
    Arguments:
        A: m by n term by document matrix
    
    Returns:
        m by 1 idf global weighting vector
    """
    A=checktype(A)
    m,N=A.shape
    dvec=zeros((N,1))
    for i in range(N):
        dvec[i,0]=A[:,i].nnz
#    print("dvec: ", dvec)
#    print("N : ", N)
    try: 
        return log2(m/dvec)
#        return (m/dvec)
    except ZeroDivisionError:
        return log2(m/(dvec+1.0))
#        return (m/(dvec+1.0))   
# G5 - Entropy: 
# Returns a column vector containing the global weight for each row of A
def entropy(A):
    """
    Entropy global weighting function as described by Brett Bader (Bader 2010). 
    Defined as entropy(A) = (1 + sum(p(i,j')*log(p(i,j'))/log(N)).  In this, N is
    the number of documents in the corpus, p(i,j') is the probability that a 
    given occurrence of i is in document j', or f(i,j') divided by the total 
    frequency of i in the corpus
    
    Best used in conjunction with the logtf local weighting.  Defined method
    logentropy does this is already implemented by this module.  If used by itself
    the returned column vector should be elementwise multiplied by each of the 
    columns of the term-document matrix
    
    Arguments:
        A: m by n term by document matrix
    
    Returns:
        m by 1 idf global weighting vector
    """
    A=checktype(A)
    dims=A.shape
    row,col=nonzero(A)
    rowsums=A.sum(1)
    vec=zeros((dims[0],1))
    for i in range(dims[0]):
        checkvec=log2(A[i,:].data/float(rowsums[i]))
        ind=isinf(checkvec)
        checkvec[ind]=0
        vec[i,0]=sum(multiply(A[i,:].data/float(rowsums[i]),checkvec))
    return 1+vec/log2(dims[1])
    
def logentropy2(A):
    """
    Function combining the logtf local weight and the entropy global weight.
    Advocated as a good weighting method for use with term-document matrices
    in Latent Semantic Analysis  (Bader 2010).
    
    Arguments:
        A: term by document matrix
    
    Returns:
        E: logentropy weighted matrix
    """
    A=checktype(A)
    globalweight=entropy(A)
    localweight=logtf(A)
    localweight=localweight.tolil()
    n=len(globalweight)
    for i in range(A.shape[1]):
        localweight[:,i]=multiply(localweight[:,i].todense(),globalweight)
    return localweight

def logentropy(A):
    """
    Function combining the log term frequency (logtf) local weight and the 
    entropy global weight.  Advocated as a good weighting method for use with 
    term-document matrices in Latent Semantic Analysis  (Bader 2010).
    
    See documentation for the logtf and entropy methods of this module for 
    further explanation of how they each work.
    
    Arguments:
        A: term by document matrix
    
    Returns:
        E: logentropy weighted matrix
    """
    A=checktype(A)
    localweight=logtf(A)
    globalweight=entropy(A)
    return dia_matrix(diag(globalweight.T[0]))*localweight

def tfidf(A):
    """
    Function combining the term-frequency (tf) local weight and the inverse
    document frequency (idf) global weight methods.
    
    See documentation for the tf and idf functions of this module for
    explanation of how each works.

    Arguments:
        A: term by document matrix
    
    Returns:
        E: tfidf weighted matrix
    """
    A=checktype(A)
#    print(type(A))
#    print(A.toarray())
#    sumtermoccurence = array(1.0/A.sum(0))[0]
#    out = np.empty_like(A)
    localweight=tf(A)
#    tf(A.toarray(),sumtermoccurence,out)
#    localweight = out
#    print("localweight\n ",localweight.toarray())
#    globalweight=idf(A)
    globalweight=idf(A)
#    print("type(localweight): ", type(localweight))
#    print("globalweight\n ",globalweight)
#    print("csc_matrix(diag(globalweight.T[0]))",csc_matrix(diag(globalweight.T[0])).toarray())
#    print("tfidf",(csc_matrix(diag(globalweight.T[0]))*localweight).toarray())
    return csc_matrix(diag(globalweight.T[0]))*localweight
#    print("globalweight.T[0] : ",np.asarray(globalweight.T[0]))
#    for i in range(localweight.shape[1]):
#        localweight[:,i] *= globalweight.T[0][i]
#    return  scipy.sparse.kron(localweight,globalweight.T[0])
#    return localweight

def pmiSpecial(A,q):
    A=checktype(A)
    dims=A.shape
    E=lil_matrix(dims)
    rowsums=A.sum(1)
    colsums=A.sum(0).T
    total=sum(rowsums)
    row,col=nonzero(A)
    for i in range(len(row)):
        E[row[i],col[i]]=float64(log2(A[row[i],col[i]]/(rowsums[row[i]]*colsums[col[i]]/total)))
    qsum=q.sum()
    weightedq=zeros(q.shape)
    qind=mlab.find(q!=0)
    for i in range(len(qind)):
        weightedq[qind[i],0]=log2(q[qind[i],0]/(rowsums[qind[i]]*qsum/total))
    return E,weightedq

def compute_weight(docs):
    print('Calculating log-entropy weights ...')	
#    tf_vectorizer = CountVectorizer(tokenizer=tokenize, stop_words=stop_words)
#    tf_vectorizer = CountVectorizer(tokenizer=tokenize)
#    docs_terms = tf_vectorizer.fit_transform(docs)
    docs = docs.todense() + 1 # add 1 to advoid NaN in log
	
    docs_terms_sum = np.sum(docs, axis=1)
    docs_prob = docs / docs_terms_sum
    log_docs_prob = np.log10(docs_prob)
    sum_entropy = (-1)*np.multiply(docs_prob, log_docs_prob)
    log_docs_num = np.log10(len(docs))
    log_entropy_docs = 1 - (sum_entropy/log_docs_num)
    entropy = np.multiply(np.log10(docs), log_entropy_docs)
    
#    print("entropy : ",entropy.shape )
    entropy_w = np.mean(entropy, axis=0)
    
    return entropy

def newidf(A):
    docs_term = np.count_nonzero(A, axis=0)
    w =[]
    for i in range(A.shape[0]):
        w1 =[]
        for j in range(A.shape[1]):
            w1.append(A.shape[0]/docs_term[j])
        w.append(w1)
    return w

def newtfidf(A):
    docs_term = np.count_nonzero(A, axis=0)
#    lendoc = np.count_nonzero(A, axis=1)
#    print(docs_term)
#    print(lendoc)
    w =[]
    for i in range(A.shape[0]):
        w1 =[]
        for j in range(A.shape[1]):
            w1.append(A[i][j] * np.log(A.shape[0]/docs_term[j] ))
        w.append(w1)
    return w

def improvednewtfidf(A,R):
#    docs_term = np.count_nonzero(A, axis=0)
#    lendoc = np.count_nonzero(A, axis=1)
#    print(docs_term)
#    print(lendoc)
    w =[]
    for i in range(A.shape[0]):
        w1 =[]
        for j in range(A.shape[1]):
            w1.append(A[i][j] * np.log(A.shape[0]/ (R[i][j]+0.001) ))
        w.append(w1)
    return np.abs(w)


def hammingpairwise_distance(A):
    hamming_distance = []
    for i in range(A.shape[0]):
        doc_distance = []
        for j in range(A.shape[0]):
            doc_distance.append(distance.hamming(A[i], A[j]))
        hamming_distance.append(doc_distance)
    return hamming_distance
