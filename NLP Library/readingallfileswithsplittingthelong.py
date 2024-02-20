# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:47:27 2019

@author: raeed
"""
import glob
import codecs
import os 
import sys
import numpy as np
sys.path.append(r'F:\Arabic dataset')


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
#        soup = BeautifulSoup(data)
#        tags = soup.find_all('text')
#        tags = soup.find_all('title')
        #store in a dictionary with ShortText Instances as values
#        instances = {}
        
        
#        for t in tags:
#            print(t)
        text = data.replace("\n", " ").replace("\r", "")
#            instances[self.my_id] = self.Text(text)
        text = self.Text(text)
#            self.my_id +=1
    
#        return instances
        return text
    
    alltext=[]
    
#    def splitter(self ,n, s):
#        pieces = s.split()
#        return (" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n))
    
    def readingfile(self, index, folder):
        txtfiles = []
        print(folder)
        count = 0
        
        for file in glob.glob(r"F:\Arabic dataset\dataset\version5\\" + folder +  r"/*.txt"):
            text = self.load_file(file)
            txtfiles.append(text)
            
            
#            if len(text.split(" "))> 50:
            text =text.split()
            text = " ".join(text[0:35])   
            
            self.alltext.append([ str(index), str(text)])
            count  += 1
#            for piece in self.splitter(20, text):
#                self.alltext.append([ str(index), str(piece)])
#                print (piece)
#            print(txtfiles)
            
#        print(alltext[1][1])
        
    def writingtofile(self,alltext):
        filefortext = open('F:\output/ArabicCorpusACTwosentence1.txt',"w",encoding="utf8")
        fileforclass = open('F:\output/ArabicCorpusACTwosentence1_label.txt',"w",encoding="utf8")
        for n in range(len(alltext)):
            if len(alltext[n][1]) < 3:
                continue
            stringtext = str(alltext[n][1])
#            print(stringtext)
            stringclass = str(alltext[n][0]) 
            filefortext.write(stringtext +"\n")
            fileforclass.write(stringclass +"\n")
        filefortext.close()
        fileforclass.close()
        
#        print(len(txtfiles))
            
    def listthefolders(self):
         my_list = os.listdir(r"F:\Arabic dataset\dataset\version5/")
         for index, folder in enumerate(my_list):
             self.readingfile(index, folder)
         np.random.shuffle(self.alltext)
         self.writingtofile(self.alltext)
#         print(my_list)
         


         
if __name__ == '__main__':
    objectname = ShortText()
    objectname.listthefolders()
    
