# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:04:15 2019

@author: Raeed
"""


# creating a new dictionary 

my_dict ={"java":100, "python":112, "c":11} 

  
# list out keys and values separately 

key_list = list(my_dict.keys()) 

val_list = list(my_dict.values()) 

print(key_list[val_list.index(100)]) 

print(key_list[val_list.index(112)]) 
 
# one-liner 

print(list(my_dict.keys())[list(my_dict.values()).index(112)]) 