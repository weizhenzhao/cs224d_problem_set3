'''
Created on 2017年10月6日

@author: weizhen
'''
import pickle
file_object=open("wordMap.bin","r")
try:
    all_the_text = file_object.read()
finally:
    file_object.close( )
output = open('wordMap2.bin', 'wb')
pickle.dump(all_the_text,output)
