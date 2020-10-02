#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:37:29 2020

@author: elsita
"""


import pandas as pd

#convert to csv
f = open('movie_lines.txt', encoding = "ISO-8859-1")
filedata = f.read()
#print(filedata)
f.close()

newdata1 = filedata.replace(",","") #replace existing commas
newdata = newdata1.replace(" +++$+++",",")
f = open("outFile.csv",'w')
f.write(newdata)
f.close()


#deal with columns
df = pd.read_csv('outFile.csv', header = None)
#remove the unnecessary cols
df.columns = ['tag', 'movie', 'movie2', 'movie3', 'pattern']
df = df[['tag', 'pattern']] 

#create response
response = df[['pattern']]
response = response.drop(response.index[len(response)-1])
response = response.reset_index(drop = True)

#remove final row in pattern
df = df.drop(df.index[0])
df = df.reset_index(drop = True)

#put back together and clean it up
final = pd.concat([df, response], axis=1, ignore_index = True)
final.columns = ['tag', 'patterns', 'responses']
final["content"] = ""

#added to decrease size of file:
final = final.loc[0:9999,:]

#write
final.to_csv("outFile2.csv", index=False)
