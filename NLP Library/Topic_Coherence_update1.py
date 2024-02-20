# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 08:28:58 2019

@author: Raeed
"""


from  nltk.tokenize import word_tokenize
import os
import re
import statistics as st

#from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

def writecoherence(coherence_array,path ):
#    print(len(vocab_array))
    filecoh = open(path ,"w",encoding="utf8")
    
    for n in range(len(coherence_array)):
        filecoh.write("Coherence: " + str(coherence_array[n] ) + "\n")
    meanchoerence = st.mean(coherence_array)
    if len(coherence_array)>1:
        std = st.stdev(coherence_array)
    else:
        std = 0 
    filecoh.write("Mean Coherence: " + str(meanchoerence ) + ", standard deviation: "+ str(std))
#    print("hi here we are! ")
    filecoh.close()
def main():
    datasets =['AlKhaleej'] #,'Akhbarona'//'AlKhaleej','AlArabiya','AlArabiya','Akhbarona'

    versions =["version3.1","version4.1","version5.1","version7.1"] #"version3.1","version3.1","version4.1","version5.1","version7.1"

    datasettypes = ['short']# ,'short', 'long'
    for datasettype in datasettypes:
        for dataset in datasets:
            for version in versions:
                          
                print(datasettype+" "+dataset +" "+ version+" ok")
#                path = 'F:\Arabic_dataset_prepared/wiki1/'+dataset+r'/'+version+'/'!
#                path = 'F:\Arabic_dataset_prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
                path = '/nuist/scratch/wangwen/wangwen_ali/raeed/Arabic_dataset_prepared/wiki/'+version+'/'
                name = 'wiki'
#                name = dataset + 'c'
                with open(path + name + '.txt' , 'r',encoding="utf8") as f:
                    text = [word_tokenize(line.replace("\n","")) for  line in f]
        
#                print(len(text))
                
#                path1 = 'F:\Arabic_dataset_prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
                path1 = '/nuist/scratch/wangwen/wangwen_ali/raeed/Arabic_dataset_prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
                
                
                dictionary = Dictionary(text)
#                my_list = os.listdir(path1)
                my_list = os.listdir(path1)
                
                for index, folder in enumerate(my_list):
                     coherencearray = []
                     if not re.findall(".txt$", folder) :
                         
                          my_list = os.listdir( path1 + folder)
                          for file in (my_list):
                               
#                              if re.findall(".topWords$", file) :
                              if re.findall(".topWords$", file) :
#                                  print(path1+folder+'/'+file)
                                  with open(path1+folder+'/'+file, 'r',encoding="utf8") as f:
                                      topics = [word_tokenize(line.replace("\n","").replace("\t"," ")) for  line in f]
                                
               
                                #cm = CoherenceModel(topics=topics, corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass')
                                  cm = CoherenceModel(topics=topics, texts=text, dictionary=dictionary, coherence='c_v',window_size =20 )
                                  coherence = cm.get_coherence()  # get coherence value
#                                  print("path1+folder: ",coherence)
                                  coherencearray.append(coherence)
                          if len(coherencearray)>0:
                              writecoherence(coherencearray,path1+folder+'/' + 'topWords.Coherencecv')
#                                text = self.load_file(folder+"/"+file, file)
#                          print(folder)
                     
#                     self.readingfile(index,dataset, version,datasettype,folder, path + folder)
#                 np.random.shuffle(self.alltext)
#                self.writingtofile(self.alltext,path,dataset,version,datasettype)
                
                
                
                
                
#                
#
#                
#                print(len(topics))
#               
#                dictionary = Dictionary(text)
#               
#                #cm = CoherenceModel(topics=topics, corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass')
#                cm = CoherenceModel(topics=topics, texts=text, dictionary=dictionary, coherence='c_v',window_size =10 )
#                coherence = cm.get_coherence()  # get coherence value
#                print(coherence)
        
if __name__ == "__main__":
    main()