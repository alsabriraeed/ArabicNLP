# coding=utf-8
import re
import string
import sys
import argparse
import script as  s
import nltk.stem.arlstem as arlstem

class arbic_clean_text:
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ«»٬¯ツ¯‘’٫٪″·'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations  
    TATWEEL = u'\u0640'
    arabic_diacritics = re.compile("""
                                 ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
    
                             """, re.VERBOSE)
    
    
    def arlstem_fun(self, text):
        obj = arlstem.ARLSTem()
        text = ''.join(obj.stem(word) for word in text)
        return text
    
    
    def normalize_arabic(self, text):
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("گ", "ك", text)
        return text
    
    def removedigist(self , text): 
        pattern = '[0-9]'
        
        text = re.sub(pattern, '', text) 
        return text
    
    def remove_diacritics(self , text):
        text = re.sub(self.arabic_diacritics, '', text)
        return text
    
    
    def remove_punctuations(self, text):
        translator = str.maketrans('', '', self.punctuations_list)
        return text.translate(translator)
    
    
    def remove_repeating_char(self, text):
        return re.sub(r'(.)\1+', r'\1', text)
    
    def removeenglisgcharacter(self, text):
        exclude = set(string.punctuation)
        text = ''.join(ch for ch in text if ch not in exclude)
        text = ''.join(list(filter(lambda x: x in string.whitespace or x not in string.printable, text)))
        return text
    
    def strip_tatweel(self, text):
        """
        Strip tatweel from a text and return a result text.
        Example:
            >>> text = u"العـــــربية"
            >>> strip_tatweel(text)
            العربية
        @param text: arabic text.
        @type text: unicode.
        @return: return a striped text.
        @rtype: unicode.
        """
        return text.replace(self.TATWEEL, '')
    
    def ar_number(self , text):
        """
        
        """
        return text.replace(s.ar_ZERO, '').replace(s.ar_ONE, '').replace(s.ar_TWO,''
                           ).replace(s.ar_THREE,'').replace(s.ar_FOUR,'').replace(s.ar_FIVE,'').replace(s.ar_SIX,''
                                    ).replace(s.ar_SEVEN,'').replace(s.ar_EIGHT,'').replace(s.ar_NINE,'')
    
    def remove_singlecharacterWord(self , text):
    
        return ' '.join( [w.replace(" ", "").replace("  ", "").replace('\u202d','').replace('\u202c','').replace('\u202e','').replace('\u200f','').replace('\u202b','')\
                          .replace('\x02','').replace('\u200d','').replace('\u200e','').replace('\ufeff','').replace('\xad','')
                          for w in text.split() if len(w.replace(" ", "").replace("  ", ""))>= 2 ] )
    
    def main(self):
        datasets =['AlKhaleej']#'ArabicAC',,'AlArabiya','Akhbarona'
#        versions =["version3","version3.1","version4",
#				"version4.1","version5","version5.1","version6","version6.1","version7","version7.1"]
#        datasets =['ArabicAC']
        versions =["version1.1"]
        for dataset in datasets:
            for version in versions:
                
                
                with open(r'F:/Arabic_dataset_prepared/'+dataset+'/'+version+'/short/'+dataset  +'c.txt', 'r',encoding="utf-8-sig") as f,\
                        open(r'F:/Arabic_dataset_prepared/'+dataset+'/'+version+'/short/'+dataset  +'c_label.txt', 'r',encoding="utf-8-sig") as fl:
                    raw_docs = [line for  line in f]
                    raw_labels = [line for  line in fl]
                print(len(raw_docs))
                print(len(raw_labels))
            #    print(s.punc_to_remove)
                text = ''
                label = ''
                file = open(r'F:/Arabic_dataset_prepared/'+dataset+'/'+version+'/short/'+dataset  +'c1.txt',"w",encoding="utf8")
                filel = open(r'F:/Arabic_dataset_prepared/'+dataset+'/'+version+'/short/'+dataset  +'c1_label.txt',"w",encoding="utf8")
                for line,linelabel in zip(raw_docs,raw_labels):
                    text = ''
                    label = ''
                    text =  line.replace("\n","") 
                    label =  linelabel.replace("\n","")
                    
            #        if len(text.split(" "))> 50:
            #            text =text.split()
            #            text = " ".join(text[0:30])
            #        here to call the arlstem 
        #            text = self.arlstem_fun(text)
                    text = self.remove_punctuations(text)
                    text = self.remove_diacritics(text)
                    text = self.remove_repeating_char(text)
                    text = self.normalize_arabic(text)
                    text = self.removeenglisgcharacter(text)
                    text = self.removedigist(text)
                    text = self.strip_tatweel(text)
                    text = self.ar_number(text)
                    text = self.remove_singlecharacterWord(text)
                    if len(text.strip()) > 2 and text.strip() != "" and  text != " ":
                        text = text + "\n"
                        label = label + "\n"
                        file.write(text)
                        filel.write(label)
                file.close()
                filel.close()
if __name__ == '__main__':
    obj = arbic_clean_text()
    obj.main()
