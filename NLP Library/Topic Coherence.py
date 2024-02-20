# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 08:28:58 2019

@author: Raeed
"""

#from palmettopy.palmetto import Palmetto
#palmetto = Palmetto()
#words = ["cake", "apple", "banana", "cherry", "chocolate"]
#palmetto.get_coherence_fast(words)
from  nltk.tokenize import word_tokenize
#from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
#from functools import wraps
#from sys import exit, stderr, stdout
#from traceback import print_exc
#
#
#def suppress_broken_pipe_msg(f):
#    @wraps(f)
#    def wrapper(*args, **kwargs):
#        try:
#            return f(*args, **kwargs)
#        except SystemExit:
#            raise
#        except:
#            print_exc()
#            exit(1)
#        finally:
#            try:
#                stdout.flush()
#            finally:
#                try:
#                    stdout.close()
#                finally:
#                    try:
#                        stderr.flush()
#                    finally:
#                        stderr.close()
#    return wrapper
#
#@suppress_broken_pipe_msg
def main():
        #with open(r'F:\Arabic_dataset_prepared/wiki/cleanedarwiki-20180920-corpus.txt', 'r',encoding="utf-8-sig") as f:
        #with open(r'F:\Arabic_dataset_prepared\wiki1\Akhbarona\version2.1/wiki.txt', 'r',encoding="utf-8-sig") as f:
        #    text = [word_tokenize(line) for  line in f]
        with open(r'F:\Arabic_dataset_prepared\ArabicAC\version2.1\long\ArabicACc.txt', 'r',encoding="utf-8-sig") as f:
            text = [word_tokenize(line) for  line in f]
        print(len(text))
        with open(r'F:\Arabic_dataset_prepared\ArabicAC\version2.1\long\bdc+9_results/ArabicACcbdc1.topWords', 'r',encoding="utf-8-sig") as f:
            topics = [word_tokenize(line) for  line in f]
        
        print(len(topics))
        #with open(r'F:\Arabic_dataset_prepared/Akhbarona/version2.1/long/vocab.txt', 'r',encoding="utf-8-sig") as f:
        #    dictionary = [word_tokenize(line) for  line in f]
        #print(len(dictionary))
        dictionary = Dictionary(text)
        #dictOfWords = { dictionary[i] : i  for i in range(0, len(dictionary) ) }
        #print(len(dictOfWords))
        #cm = CoherenceModel(topics=topics, corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass')
        cm = CoherenceModel(topics=topics, texts=text, dictionary=dictionary, coherence='c_npmi',window_size =10 )
        coherence = cm.get_coherence()  # get coherence value
        print(coherence)
        
if __name__ == "__main__":
    main()