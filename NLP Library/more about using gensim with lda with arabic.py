# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:37:22 2019

@author: Raeed
"""

from gensim.corpora import Dictionary
from gensim.models import ldamodel
import numpy as np
import os
import glob
import codecs

class gensimforarabic:
    
    def Text(self, text):
        '''
        For printing purposes.
        '''
        return  text
    
    
    def load_file(self,filename):
        #retrieve the original text 
        with codecs.open(filename, encoding='utf-8') as f:
            data = f.read()

        text = data.replace("\n", " ").replace("\r", "")
        text = self.Text(text)
        
        return text.split()
    
    alltext=[]
    def readingfile(self, index, folder):
#        txtfiles = []
#        print(folder)
        count = 0
        
        for file in glob.glob(r"F:\Arabic dataset\dataset\version5\\" + folder +  r"/*.txt"):
            text = self.load_file(file)
#            txtfiles.append(text)
#            self.alltext.append([ str(index), str(text)])
            self.alltext.append(text)
            count  += 1
        
        
        
    def LDAmodel(self, alltext):
#        print(alltext[0][0])
        dictionary = Dictionary(alltext)
        corpus = [dictionary.doc2bow(text) for text in alltext]
#        print(len(dictionary))
#        print(corpus[0])
        np.random.seed(1) # setting random seed to get the same results each time.
        model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics= 9,iterations=500)
#        print(model.show_topics())
#        print(model.get_term_topics('عمل'))
#        print(model.phi_values)
        print(model.show_topics(num_topics=9,num_words=9))
            
    
    def listthefolders(self):
         my_list = os.listdir(r"F:\Arabic dataset\dataset\version5/")
         for index, folder in enumerate(my_list):
             self.readingfile(index, folder)
         np.random.shuffle(self.alltext)
#         self.writingtofile(self.alltext)
         self.LDAmodel(self.alltext)
#         print(my_list)
    
    
    
if __name__ == '__main__':
    objectname = gensimforarabic()
    objectname.listthefolders()