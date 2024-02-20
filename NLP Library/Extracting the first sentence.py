# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 17:55:05 2019
@author: raeed
"""
import sys
from nltk.tokenize import sent_tokenize
sys.path.append(r'F:\output')
import nltk
class CreatingShortText:
    
    def main(self):
        datasets =['AlArabiya'] #,'Akhbarona','AlArabiya','Akhbarona','ArabicAC',
        versions =["version1"]#,"version3.1",		"version4.1","version5.1","version7.1"   
        datasettypes = ['long']# ,'short', 'long'
        for datasettype in datasettypes:
            for dataset in datasets:
                for version in versions:
                          
                    print(datasettype+" "+dataset +" "+ version+" ok")
        #            path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/'
                #    path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/'
                    path = 'F:\Arabic_dataset_prepared\\'+dataset+r'\\'+version+'\\'+datasettype+'\/'
#                    path = '/home1/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
                    path1 = 'F:\Arabic_dataset_prepared\\'+dataset+r'\\'+version+'\\short\/'
                    name = dataset
                    with open(path + name +'c.txt', 'r',encoding="utf8") as f:
                        
#                         raw_docs = [sent_tokenize(line)[0]  for  line in f]
                        raw_docs = [line.split(".")[0]  for  line in f]
                    
#                         bag_of_words = raw_docs               
#                         tokenized_word = [[words for words in nltk.word_tokenize(bag_of_words[i])]
#                                                for i in range(len(bag_of_words)) ]
                    print(len(raw_docs))
                    
                    file = open(path1 + name+'c.txt',"w",encoding="utf8")
                    for line in range(len(raw_docs)):
                        str1 = ""
                        str1 = raw_docs[line] 
                        file.writelines(str1 +"\n" ) 
#                               break
                           
                    file.close()
                                       
                    
if __name__ == '__main__':
    obj = CreatingShortText()
    obj.main()