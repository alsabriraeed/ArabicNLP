# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:20:16 2019

@author: Raeed
"""
import glob
import nltk
import numpy as np
import pandas as pd
import os.path
import codecs

from csv import DictWriter
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS #, Arabic_STOP_WORDS
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
stop_words = set(stopwords.words('arabic'))#.union(set(Arabic_STOP_WORDS))
#word_list = set(nltk_words.words())
vocabulary = []
vocab_index = {}
vocab_size = 0
doc_size = 0

WEIGHT_CSV = ('weight.csv')
WEIGHT_CSV_SORT = ('weight_sort.csv')
EXPECTED_TERM_NUM = 50

# Token lemmatizing
def lemmatize_tokens(tokens, wordnet_lemmatizer):
    lemmatized = []
    for item in tokens:
        lemmatized.append(wordnet_lemmatizer.lemmatize(item))
    return lemmatized

# Token stemming
def stem_tokens(tokens, porter_stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(porter_stemmer.stem(item))
    return stemmed
	
# Build a vocabulary
def tokenize(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    lemma = lemmatize_tokens(tokens, wordnet_lemmatizer)
    port_stem = stem_tokens(lemma, porter_stemmer)
	# Build a list of words ignoring stop worss and illegal characters
#    filtered_stems = filter(lambda stem: stem not in stop_words and stem.isalpha() and stem in word_list, port_stem)   
#    return filtered_stems
    return port_stem


def compute_weight(docs):
    print('Calculating log-entropy weights ...')	
#    tf_vectorizer = CountVectorizer(tokenizer=tokenize, stop_words=stop_words)
    tf_vectorizer = CountVectorizer(tokenizer=tokenize)
    docs_terms = tf_vectorizer.fit_transform(docs)
    docs_terms = docs_terms.todense() + 1 # add 1 to advoid NaN in log
	
    docs_terms_sum = np.sum(docs_terms, axis=1)
    docs_prob = docs_terms / docs_terms_sum
    log_docs_prob = np.log10(docs_prob)
    sum_entropy = (-1)*np.multiply(docs_prob, log_docs_prob)
    log_docs_num = np.log10(doc_size)
    log_entropy_docs = 1 - (sum_entropy/log_docs_num);
    entropy = np.multiply(np.log10(docs_terms), log_entropy_docs)
    
    print("entropy : ",entropy.shape )
    entropy_w = np.mean(entropy, axis=0) 

    print('Calculating TF-IDF weights ...')
    words_idf = np.zeros(vocab_size)
    for text in docs:
        words = set(tokenize(text))
        indexes = [vocab_index[word] for word in words] # Mapping docs with index of vocabulary
        #print(indexes)
        words_idf[indexes] += 1.0
    print(words_idf)
    print(len(words_idf))
    idf_w = np.log(doc_size / (1 + words_idf).astype(float))

    print('Calculating TDV weights ...')	
    words_tdv = np.zeros((doc_size, vocab_size))
#    tf_vectorizer = CountVectorizer(tokenizer=tokenize, stop_words=stop_words)
    tf_vectorizer = CountVectorizer(tokenizer=tokenize)
#    tf_vectorizer = CountVectorizer(tokenizer=tokenize, stop_words=stop_words)
    print(np.shape(tf_vectorizer))
#    print(tf_vectorizer)
#    print(docs)
    input("Press Enter to continue...")
    docs_terms = tf_vectorizer.fit_transform(docs)
    docs_terms = docs_terms.todense()
	
	# Initialize a centroid as a reference for similarity computation
    centroid = np.mean(docs_terms, axis=0)
    row, columns = docs_terms.shape
    print(columns)
#    for i in range(doc_size):
    for i in range(doc_size):
#        print(i)
        di = docs_terms[i]
        di_dist = np.linalg.norm(centroid - di)	# AvgSim
#        for k in range(vocab_size):
        for k in range(columns):
            if di[0, k] != 0:	# If document contains term
                di_k = np.array(di)
                di_k[0, k] = 0	# Then remove this term from document
                di_k_dist = np.linalg.norm(centroid - di_k)	# Recalculate distance
            else:
                di_k_dist = di_dist
            words_tdv[i, k] = di_dist - di_k_dist

    tdv_w = np.mean(words_tdv, axis=0)
	
    return entropy_w, tdv_w, idf_w

def export_to_file(idf_weights, tdv_weights, entropy_weights, filename):
    entropy_arr = np.squeeze(np.asarray(entropy_weights))
    with open(filename, 'w', newline='',encoding="utf8") as csvfile:
        fieldnames = ['term', 'idf weight', 'tdv weight', 'entropy weight']
        writer = DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(vocab_size):
            writer.writerow({
                'term': vocabulary[i],
                'idf weight': idf_weights[i],
                'tdv weight': tdv_weights[i],
                'entropy weight': entropy_arr[i]
            })
    print('Weight result has been saved to', filename)
	
def export_to_file_sort(idf_weights, tdv_weights, entropy_weights, filename):
    entropy_arr = np.squeeze(np.asarray(entropy_weights))
    idf_term_list = pd.Series(idf_weights)
    idf_w_des = idf_term_list.sort_values(ascending=False)
    idf_w_des_idx = idf_w_des.index
    
    tdv_term_list = pd.Series(tdv_weights)
    tdv_w_des = tdv_term_list.sort_values(ascending=False)
    tdv_w_des_idx = tdv_w_des.index	
    
    entropy_term_list = pd.Series(entropy_arr)
    entropy_w_asc = entropy_term_list.sort_values(ascending=True)
    entropy_w_asc_idx = entropy_w_asc.index		
    with open(filename, 'w', newline='',encoding="utf8") as csvfile:
        fieldnames = ['idf_term_sort', 'idf_weight_sort','tdv_term_sort', 'tdv_weight_sort','entropy_term_sort', 'entropy_weight_sort']
        writer = DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(vocab_size):
            writer.writerow({
                'idf_term_sort':vocabulary[idf_w_des_idx[i]],
                'idf_weight_sort': idf_w_des.values[i],
                'tdv_term_sort':vocabulary[tdv_w_des_idx[i]],
                'tdv_weight_sort': tdv_w_des.values[i],	
                'entropy_term_sort':vocabulary[entropy_w_asc_idx[i]],
                'entropy_weight_sort': entropy_w_asc.values[i],					
            })
    print('Weight result has been saved to', filename)

def analyze_term_weight(idf_weights, tdv_weights, entropy_weights):
    entropy_arr = np.squeeze(np.asarray(entropy_weights))
    term_list = pd.Series(entropy_arr)
    smallest = term_list.nsmallest(EXPECTED_TERM_NUM)
    print(smallest)
    print('Log-entropy: 50 most important terms');
    for i in smallest.index:
        print(vocabulary[i])
    input("Press Enter to continue...")
    
    term_list = pd.Series(idf_weights)
    largest = term_list.nlargest(EXPECTED_TERM_NUM)
    print(largest)
    print('IDF: 50 most important terms');
    for i in largest.index:
        print(vocabulary[i])
    input("Press Enter to continue...")
    
    term_list = pd.Series(tdv_weights)
    largest = term_list.nlargest(EXPECTED_TERM_NUM)
    print(largest)
    print('TDV: 50 most important terms');
    for i in largest.index:
        print(vocabulary[i])
    input("Press Enter to continue...")
	
def main():
#    corpus = input("Press Enter to work on all corpora in directory OR input the corpus dir name you want to work on:")
    #print(corpus)
#    corpus_name = corpus_name = '20news-bydate/' + corpus;
#    corpus_path = '20news-bydate/' + corpus +'/*'
#    
    corpus_name = corpus_name = r'C:\Users\Raeed\workspace\STTM\STTM\corpusforweight/ArabicCorpusACTwosentence1.txt'
    corpus_path = r'C:\Users\Raeed\workspace\STTM\STTM\corpusforweight/ArabicCorpusACTwosentence1.txt'
    #print(corpus_path)
    if False == os.path.exists(corpus_name):
        print('Can not find corpus directory', corpus_name)
        return
	
#    if not corpus:
#        corpus_path = '20news-bydate/*/*';
	
    files = glob.glob(corpus_path)	# Read all file objects
    print(files)
    docs = []
    doc_ids = []
    ids =0
    for file in files:
        with codecs.open(file, 'r',encoding="utf8") as f:
            docs = [line.replace("\n","") for  line in f]
#            docs.append(f.read())   # append content of file
#            doc_ids.append(file)    # append file name to list
            doc_ids.append(ids)    # append file name to list
            ids+=1
    global doc_size
    doc_size = len(docs)
    print('Total documents:', doc_size)
	
    vocab = set()
    for text in docs:   # Text has the whole content of a file name
        #print(words)
        #input("Press Enter to continue...")
        words = tokenize(text)  # Normalize -> tokenize -> lemmatize -> stem -> stop words 
        vocab.update(words)		# 

    global vocabulary
    vocabulary = sorted(list(vocab))
    global vocab_size
    vocab_size = len(vocabulary)
    global vocab_index
    vocab_index = {w: idx for idx, w in enumerate(vocabulary)}  # Bind vocab entries to indexes
    print('Vocabulary size:', vocab_size)
    # Calculate weights for variations
    entropy_weights, tdv_weights, idf_weights = compute_weight(docs)
    # Import weights to csv file
    export_to_file(idf_weights, tdv_weights, entropy_weights, WEIGHT_CSV)
    export_to_file_sort(idf_weights, tdv_weights, entropy_weights, WEIGHT_CSV_SORT)
    analyze_term_weight(idf_weights, tdv_weights, entropy_weights)


if __name__ == '__main__':
    main()

