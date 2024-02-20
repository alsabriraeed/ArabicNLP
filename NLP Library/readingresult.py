# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:47:27 2019

@author: raeed
"""
import glob
import codecs
import os 
import sys
import re
import csv
import numpy as np
#sys.path.append(r'F:\Arabic dataset prepared/')
#sys.path.append('/home1/raeed/Arabic dataset prepared/')
from  nltk.tokenize import word_tokenize


class ShortText:
#    def __init__(self, my_id, text):
#        self.id = my_id         
#        self.text = text
    my_id = 0    
    def Text(self, text):
        '''
        For printing purposes.
        '''
        return  text

    def load_file(self,filename, file):
        #retrieve the original text 
        with codecs.open(filename, encoding='utf-8') as f:
            data = f.readlines()
#        print(data)
#        print (data[len(data)-1])
#        for line in data:
#            print(line +"hhhhhhhhhh")
        print(filename+ file)
#        text = data.replace("\n", " ").replace("\r", "  ")
        if file == 'theta.accuracy'  or file == 'topWords.Coherence':
            text = data[len(data)-1]
        elif  file == 'theta.PurityNMI':
            text = data[len(data)-1] +" "+ data[len(data)-2]
            
        text= word_tokenize(str(text))
#        print(text)
#            instances[self.my_id] = self.Text(text)
#        text = self.Text(text)
#            self.my_id +=1
    
#        return instances
        return text
    
    alltext=[]
    str1 = 'Model_Name'+'|'+'Accuracy'+'|'+'Acc_Std'+'|'+ 'Purity'+'|'+'Purity_Std'+'|'+'NMI'+'|'+'NMI_Std'+'|'+'Coherence'+ '|'+'Coherence_Std'
    
    def readingfile(self, index,dataset, version,datasettype,modelname,folder):
        
        result = modelname+'|'
#        print(folder)
        
#        print( folder)
        my_list = os.listdir(folder)
        for file in (my_list):
            if file == 'theta.accuracy' or file == 'theta.PurityNMI' or file == 'topWords.Coherence':
#                print(folder+file)
                text = self.load_file(folder+"/"+file, file)
                if file == 'theta.accuracy':
                    result+=  text[3]+'|'+text[8]+'|'
                elif file == 'theta.PurityNMI':
                    result+=  text[12]+'|'+ text[17]+'|'+ text[3]+'|'+text[8]
                elif file == 'topWords.Coherence':
                    result +=  text[3]+'|'+text[8]+'|'
        self.alltext.append(result)
#        for file in glob.glob(folder + r"/*.txt"):
#            print("kkkkkkkk")
#            text = self.load_file(file)
#            txtfiles.append(text)
#            self.alltext.append([str(index), str(text)])
#            count  += 1
        
#            print(txtfiles)
            
#        print(alltext[1][1])
        
    def writingtofile(self,alltext,path,dataset,version,datasettype):
        filefortext = open(path+dataset+'_'+ version+'_'+datasettype +'.txt',"w",encoding="utf8")
        for n in range(len(alltext)):
            
            stringtext = str(alltext[n])
#            print(stringtext)
            filefortext.write(stringtext +"\n")
        filefortext.close()
        
#        print(len(alltext))
        
            
    def listthefolders(self):
         datasets =['ArabicAC','AlKhaleej','AlArabiya','Akhbarona']#,'AlKhaleej','AlArabiya','Akhbarona'
         versions =["version2.1","version3.1",
				"version4.1","version5.1","version7.1"]
#         datasets =['Akhbarona']
         datasettype='long' # 'long'
         for dataset in datasets:
             for version in versions:
                 self.alltext.clear()
                 self.alltext.append(self.str1)
#                 path = 'F:\Arabic dataset prepared/'+dataset+'/'+version+'/'+datasettype+'/'
#                 path = '/home1/raeed/Arabic dataset prepared/'+dataset+'/'+version+'/'+datasettype+'/'
                 path = '/nuist/scratch/wangwen/wangwen_ali/raeed/Arabic_dataset_prepared/'+dataset+r'/'+version+'/'+datasettype+'/'

                 my_list = os.listdir(path)
                 for index, folder in enumerate(my_list):
                     if not re.findall(".txt$", folder) :
#                         print(folder)
                         
                         self.readingfile(index,dataset, version,datasettype,folder, path + folder)
#                 np.random.shuffle(self.alltext)
                     self.writingtofile(self.alltext,path,dataset,version,datasettype)
             
#                 print(my_list)
         
if __name__ == '__main__':
    objectname = ShortText()
    objectname.listthefolders()
    
