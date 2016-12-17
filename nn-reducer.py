#!/home/shivam/anaconda3/bin/python

import sys
import pickle
import ast
import numpy as np

file_dir = "/home/shivam/Documents/HadoopStreaming/"

with open(file_dir+"fileweights/w0.txt","rb") as f:
    w0= np.array(pickle.load(f), dtype=np.float32)
with open(file_dir+"fileweights/w1.txt","rb") as f:
    w1 = np.array(pickle.load(f), dtype=np.float32)
with open(file_dir+"fileweights/w2.txt","rb") as f:
    w2 = np.array(pickle.load(f), dtype=np.float32)
with open(file_dir+"fileweights/b0.txt","rb") as f:
    b0= pickle.load(f)
with open(file_dir+"fileweights/b1.txt","rb") as f:
    b1 = pickle.load(f)
with open(file_dir+"fileweights/b2.txt","rb") as f:
    b2 = pickle.load(f)


count2=0
count1=0
count0=0
er=0

alpha=0.1

for line in sys.stdin:
			key,value=line.split(': ')
			value = np.array(ast.literal_eval(value.strip()), dtype=np.float32)
			if key=='w2':
				count2=count2+value
				#w2-=alpha*(np.sum(errors, axis = 0)/12.)
			if key=='w1':
				count1=count1+value
				#w1-=alpha*(np.sum(errors, axis = 0)/12.)
			if key=='w0':
				count0=count0+value
				#w0-=alpha*(np.sum(errors, axis = 0)/12.)
			if key=='loss':
				er=er+value
				#print key, np.sum(errors)/12.0


w2-=alpha*(count2/12.)
w1-=alpha*(count1/12.)
w0-=alpha*(count0/12.)



with open(file_dir+"fileweights/w0.txt", "wb") as f:
	pickle.dump(w0, f)

with open(file_dir+"fileweights/w1.txt", "wb") as f:
    pickle.dump(w1, f)

with open(file_dir+"fileweights/w2.txt", "wb") as f:
    pickle.dump(w2, f)


with open(file_dir+"fileweights/b0.txt", "wb") as f:
    pickle.dump(b0, f)


with open(file_dir+"fileweights/b1.txt", "wb") as f:
    pickle.dump(b1, f)


with open(file_dir+"fileweights/b2.txt", "wb") as f:
    pickle.dump(b2, f)

print("loss: ", er/12.)