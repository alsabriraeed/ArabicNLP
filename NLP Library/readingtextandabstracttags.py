# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 08:40:19 2019

@author: Raeed
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:47:27 2019

@author: raeed
"""
import glob
import codecs
from bs4 import BeautifulSoup   
import os 
import sys
import numpy as np
import re
sys.path.append(r'F:\antcorpus.data-master')


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

    def load_file(self,filename):
        #retrieve the original text 
        with codecs.open(filename, encoding='utf-8') as f:
            data = f.read()
        #use beautifulsoup to get tag attributes and elements
        soup = BeautifulSoup(data)
#        tags = soup.find_all('text')
#        tags = soup.find_all(['abstract', 'title'])
#        tags = soup.find_all(['abstract', 'title'])
#        tags = soup.findAll(re.compile(r'(abstract|title)'))
        tags = soup.findAll(re.compile(r'(abstract)'))
#        tags = soup.title
#        print(tags)
        
        #store in a dictionary with ShortText Instances as values
#        instances = {}
        
        
#        for t in tags:
        for t in tags:
            text = t.get_text().replace("\n", "").replace(" ", " ").strip()
#            instances[self.my_id] = self.Text(text)
            text = self.Text(text)
#            self.my_id +=1
    
#        return instances
            return text
    
    alltext=[]
    def readingfile(self, index, folder):
        txtfiles = []
#        print(folder)
        count = 0
        
        for file in glob.glob(r"F:\Arabic dataset\AlArabiya\\" + folder +  r"/*.txt"):
            text = self.load_file(file)
            if text == "":
                continue
            txtfiles.append(text)
            self.alltext.append([ str(index), str(text)])
            count  += 1
#        print(self.alltext[1][1])
#            print(txtfiles)
            
      
        
    def writingtofile(self,alltext):
        filefortext = open('F:\output/AlArabiya/AlArabiya.txt',"w",encoding="utf8")
        fileforclass = open('F:\output/AlArabiya/AlArabiya_label.txt',"w",encoding="utf8")
        print(len(alltext))
        for n in range(len(self.alltext)):
#        for n in range(9900):
            stringtext = str(self.alltext[n][1])
            if stringtext.strip() == "" or stringtext == " " or len(stringtext.split())<3:
                continue
#            print(stringtext)
            stringclass = str(self.alltext[n][0]) 
            filefortext.write(stringtext +"\n")
            fileforclass.write(stringclass +"\n")
        filefortext.close()
        fileforclass.close()
        
#        print(len(txtfiles))
            
    def listthefolders(self):
         my_list = os.listdir(r"F:\Arabic dataset\AlArabiya/")
         for index, folder in enumerate(my_list):
             self.readingfile(index, folder)
         np.random.shuffle(self.alltext)
#         print(self.alltext[0][1])
         self.writingtofile(self.alltext)
         print(my_list)
         
        
        
        

if __name__ == '__main__':
    objectname = ShortText()
    objectname.listthefolders()
    
