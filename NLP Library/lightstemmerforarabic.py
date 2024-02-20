# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 00:45:44 2019

@author: raeed
"""


from nltk.stem.isri import ISRIStemmer
import sys
import nltk
sys.path.append(r'F:\output')



# Arabic light stemming for Arabic text
# takes a word list and perform light stemming for each Arabic words
def load_stop_words():
    path = r'F:\for master\MS CS\relating to research\new papers\more related to my work\process-arabic-text-master\stop_words.txt'
    with open(path, 'r',encoding="utf8") as f:
        raw_docs = [line.replace('\n','')  for  line in f]
    stop_words =[]
    for word in raw_docs:
        stop_words.append(word)
    
    return stop_words

def light_stem(text):
#    words = text.split()
    words = text
    
    result = list()
    stemmer = ISRIStemmer()
    stopwords = stemmer.stop_words+load_stop_words()
#    print(stemmer.stop_words)
    for word in nltk.word_tokenize(words):
#        print(word)
        word = stemmer.norm(word, num=1)      # remove diacritics which representing Arabic short vowels
        if not word in stopwords:    # exclude stop words from being processed
#            print("you are  in ")
#            print(word)
            word = stemmer.pre32(word)        # remove length three and length two prefixes in this order
            word = stemmer.suf32(word)        # remove length three and length two suffixes in this order
            word = stemmer.waw(word)          # remove connective ‘و’ if it precedes a word beginning with ‘و’
            word = stemmer.norm(word, num=2)  # normalize initial hamza to bare alif
            result.append(word)
#    print(result)
    return ' '.join(result)


def readingarticles():
    file = open(r'F:\Arabic dataset prepared\ArabicAC\version9.1/short/exampleforstemmerslight10tstem.txt',"w",encoding="utf8")
    with open(r'F:\Arabic dataset prepared\ArabicAC\version9.1/short/exampleforstemmers.txt', 'r',encoding="utf8") as f:
        raw_docs = [line for  line in f]
    text =''
#    print(bag_of_words[0])
    for line in raw_docs:
        
        text =  light_stem(line.replace("\n","")) +"\n"
#        print(text)
        file.write(text)
    
#    file.colse()
    
    
if __name__ == '__main__':
    readingarticles()

