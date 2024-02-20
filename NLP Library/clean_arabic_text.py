# coding=utf-8
import re
import string
import sys
import argparse

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ«»٬'''
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


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def removedigist(text): 
    pattern = '[0-9]'
    text = re.sub(pattern, '', text) 
    return text

def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def removeenglisgcharacter(text):
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    text = ''.join(list(filter(lambda x: x in string.whitespace or x not in string.printable, text)))
    return text

def strip_tatweel(text):
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
    return text.replace(TATWEEL, '')

def remove_singlecharacterWord(text):

    return ' '.join( [w for w in text.split() if len(w)>2] )

if __name__ == '__main__':
    file = open(r'F:\output\arabicwiki\AlArabiyaarlstem.txt',"w",encoding="utf8")
    with open(r'F:\output\arabicwiki\AlArabiya.txt', 'r',encoding="utf8") as f:
        raw_docs = [line for  line in f]
    text =''
    for line in raw_docs:
        
        text =  line.replace("\n","")  
        
#        if len(text.split(" "))> 50:
#            text =text.split()
#            text = " ".join(text[0:30])
        text = remove_punctuations(text)
        text = remove_diacritics(text)
        text = remove_repeating_char(text)
        text = normalize_arabic(text)
        text = removeenglisgcharacter(text)
        text = removedigist(text)
        text = strip_tatweel(text)
        text = remove_singlecharacterWord(text)
        
        text = text + "\n"
        file.write(text)
    file.close()
