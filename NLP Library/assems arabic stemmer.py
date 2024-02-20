# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:10:55 2019

@author: Raeed
"""

"""

"""


from arabicstemmer import ArabicStemmer
class ARLstemCalling:

    def arlstem_fun(self, text):
        stemmer = ArabicStemmer()
        text = ''.join(stemmer.stemWord(word) for word in text)
        return text

    def main(self):
        file = open(r'F:\Arabic dataset prepared\Akhbarona\version6.1/long/Akhbarona.txt', "w", encoding="utf8")
        filel = open(r'F:\Arabic dataset prepared\Akhbarona\version6.1/long/Akhbarona_label.txt', "w", encoding="utf8")
        with open(r'F:\Arabic dataset prepared\Akhbarona\version2.1/long/Akhbaronac.txt', 'r', encoding="utf-8-sig") as f, open(
                r'F:\Arabic dataset prepared\Akhbarona\version2.1/long/Akhbaronac_label.txt', 'r', encoding="utf-8-sig") as fl:
            raw_docs = [line for line in f]
            raw_labels = [line for line in fl]
        print(len(raw_docs))
        print(len(raw_labels))
        #    print(s.punc_to_remove)
        text = ''
        label = ''
        for line, linelabel in zip(raw_docs, raw_labels):
            text = ''
            label = ''
            text = line.replace("\n", "")
            label = linelabel.replace("\n", "")

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
            if len(text.strip()) > 2 and text.strip() != "" and text != " ":
                text = text + "\n"
                label = label + "\n"
                file.write(text)
                filel.write(label)
        file.close()
        filel.close()


if __name__ == '__main__':
    obj = ARLstemCalling()
    obj.main()
