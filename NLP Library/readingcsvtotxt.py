# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:29:10 2020

@author: Raeed
"""

import csv
csv_file =r'F:/ArabicDatasets/DataSet for Arabic Classification/arabic_dataset_classifiction.csv/arabic_dataset_classifiction.csv'
txt_file = r'F:/ArabicDatasets/DataSet for Arabic Classification/arabic_dataset_classifiction.csv/arabic_dataset_classifiction.txt'
label_file = r'F:/ArabicDatasets/DataSet for Arabic Classification/arabic_dataset_classifiction.csv/arabic_dataset_classifiction_label.txt'

with open(txt_file, "w",encoding="utf8") as my_output_file, open(label_file, "w",encoding="utf8") as my_label_file:
    with open(csv_file, "r",encoding="utf8") as my_input_file:
        for row in csv.reader(my_input_file):
            if len(row[0]) >0:
                my_output_file.write(row[0]+'\n')
                my_label_file.write(row[1]+'\n')
#            print(row[0])
#        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()