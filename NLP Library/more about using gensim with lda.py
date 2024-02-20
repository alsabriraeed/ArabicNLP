# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 07:55:31 2019

@author: Raeed
"""
from gensim.corpora import Dictionary
from gensim.models import ldamodel
import numpy


texts = [['bank','river','shore','water'],
        ['river','water','flow','fast','tree'],
        ['bank','water','fall','flow'],
        ['bank','bank','water','rain','river'],
        ['river','water','mud','tree'],
        ['money','transaction','bank','finance'],
        ['bank','borrow','money'], 
        ['bank','finance'],
        ['finance','money','sell','bank'],
        ['borrow','sell'],
        ['bank','loan','sell']]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print(dictionary)
print(corpus)

