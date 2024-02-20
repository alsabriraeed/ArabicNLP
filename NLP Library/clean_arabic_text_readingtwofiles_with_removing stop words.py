# coding=utf-8
import re
import string
import sys
import argparse
import script as  s
import nltk.stem.arlstem as arlstem
from nltk.stem.isri import ISRIStemmer
class arbic_clean_text:
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ«»٬¯ツ¯‘’٫٪″'''
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
    
    def load_stop_words(self):
        path = r'F:\for master\MS CS\relating to research\new papers\more related to my work\process-arabic-text-master\stop_words.txt'
        with open(path, 'r',encoding="utf8") as f:
            raw_docs = [line.replace('\n','')  for  line in f]
        stop_words =[]
        for word in raw_docs:
            stop_words.append(word)
    
        return stop_words

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
        stemmer = ISRIStemmer()
        stopwords = stemmer.stop_words+self.load_stop_words()
        return ' '.join( [w.replace(" ", "").replace("  ", "").replace('\u202d','').replace('\u202c','').replace('\u202e','').replace('\u200f','').replace('\u202b','')\
                          .replace('\x02','').replace('\u200d','').replace('\u200e','')
                          for w in text.split() if len(w.replace(" ", "").replace("  ", ""))>= 2 and not w in stopwords ] )
    
    def main(self):
        path = r'F:\ArabicDatasets/Akhbaronashort/version1/short/'
        file = open(path+'Akhbaronac.txt',"w",encoding="utf8")
        filel = open(path+'Akhbaronac_label.txt',"w",encoding="utf8")
        with open(path+'Akhbarona.txt', 'r',encoding="utf-8-sig") as f,\
        open(path+'Akhbarona_label.txt', 'r',encoding="utf-8-sig") as fl:
            raw_docs = [line for  line in f]
            raw_labels = [line for  line in fl]
        print(len(raw_docs))
        print(len(raw_labels))
    #    print(s.punc_to_remove)
        text = ''
        label = ''
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
            if len(text.strip()) > 3 and text.strip() != "" and  text != " ":
                text = text + "\n"
                label = label + "\n"
                file.write(text)
                filel.write(label)
        file.close()
        filel.close()
if __name__ == '__main__':
    obj = arbic_clean_text()
    obj.main()
