# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:10:55 2019

@author: Raeed
"""

"""
ARLSTem Arabic Stemmer
The details about the implementation of this algorithm are described in:
K. Abainia, S. Ouamour and H. Sayoud, A Novel Robust Arabic Light Stemmer ,
Journal of Experimental & Theoretical Artificial Intelligence (JETAI'17),
Vol. 29, No. 3, 2017, pp. 557-573.
The ARLSTem is a light Arabic stemmer that is based on removing the affixes
from the word (i.e. prefixes, suffixes and infixes). It was evaluated and
compared to several other stemmers using Paice's parameters (under-stemming
index, over-stemming index and stemming weight), and the results showed that
ARLSTem is promising and producing high performances. This stemmer is not
based on any dictionary and can be used on-line effectively.
"""
import nltk.stem.arlstem as arlstem
from nltk.tokenize import word_tokenize
import ARLSTem
class ARLstemCalling:
    
#    def arlstem_fun(self, text):
#        obj = arlstem.ARLSTem()
#        text = ''.join(obj.stem(word) for word in text)
#        print(text)
#        return text
    def arlstem_fun(self, text):
        obj = ARLSTem.ARLSTem()
#        print(' '.join(obj.stem(word) for word in word_tokenize(text)))
        text = (' '.join(obj.stem(word) for word in word_tokenize(text)))
#        print(text)
        return text
    
    def main(self):
        datasets =['ArabicAC','AlKhaleej','AlArabiya','Akhbarona'] #,'Akhbarona'
        versions =["version2.1"]
    
        datasettypes = ['short']# ,'short', 'long'
        for datasettype in datasettypes:
            for dataset in datasets:
                for version in versions:
                                
                    print(datasettype+" "+dataset +" "+ version+" ok")
        #            path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/'
                #    path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/'
                    path = 'E:\Arabic_dataset_prepared\\'+dataset+r'\\'+version+'\\'+datasettype+'\/'
#                    path = '/home1/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
                    path1 = 'E:\Arabic_dataset_prepared\\'+dataset+r'\\version3.1\\'+datasettype+'\/'
                    name = dataset
                    file = open(path1+ name +'c.txt',"w",encoding="utf-8-sig")
                    with open(path+name +'c.txt', 'r',encoding="utf-8-sig") as f:
                        raw_docs = [line for  line in f]
                    print(len(raw_docs))
                #    print(s.punc_to_remove)
                    text = ''
                    for line in raw_docs:
                        text = ''
                        text =  line.replace("\n","") 
                        
                #        if len(text.split(" "))> 50:
                #            text =text.split()
                #            text = " ".join(text[0:30])
                #        here to call the arlstem 
                    
                        text = self.arlstem_fun(text)
            #            text = self.remove_punctuations(text)
            #            text = self.remove_diacritics(text)
            #            text = self.remove_repeating_char(text)
            #            text = self.normalize_arabic(text)
            #            text = self.removeenglisgcharacter(text)
            #            text = self.removedigist(text)
            #            text = self.strip_tatweel(text)
            #            text = self.ar_number(text)
            #            text = self.remove_singlecharacterWord(text)
                        if len(text.strip()) > 2 and text.strip() != "" and  text != " ":
                            text = text + "\n"
                            file.write(text)
                    file.close()
                    
if __name__ == '__main__':
    obj = ARLstemCalling()
    obj.main()
