#!/home/shivam/anaconda3/bin/python

import pickle
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
import sys
from json import dumps
	
alpha=0.01

file_dir = "/home/shivam/Documents/HadoopStreaming/"
	
with open(file_dir+"fileweights/w0.txt","rb") as f:
    w0= pickle.load(f)
with open(file_dir+"fileweights/w1.txt","rb") as f:
    w1 = pickle.load(f)
with open(file_dir+"fileweights/w2.txt","rb") as f:
    w2 = pickle.load(f)
with open(file_dir+"fileweights/b0.txt","rb") as f:
    b0= pickle.load(f)
with open(file_dir+"fileweights/b1.txt","rb") as f:
    b1 = pickle.load(f)
with open(file_dir+"fileweights/b2.txt","rb") as f:
    b2 = pickle.load(f)
	
def tanh(x, deriv=False):
	if deriv==True:
		return 1-(np.tanh(x)**2)
	return np.tanh(x)			
for line in sys.stdin:
	data = line.split(',')
	X = np.array(data[:-1], dtype=np.float64).reshape(-1,1)
	Y=float(data[-1])	
	X=np.reshape(X,(54675,1))

			
	#feed forward
	z1 = (np.dot(X.T, w0))+b0
	a1 = np.tanh(z1)
	z2 = (np.dot(a1, w1))+b1
	a2 = np.tanh(z2)
	z3 = (np.dot(a2, w2))+b2
	a3 = np.tanh(z3)
		
	#loss
	loss = np.sum(0.5*(Y  - a3)**2)
		        		
	#backpropagation    
	dw2 = (a3 - Y) * tanh(z3,deriv=True)
	dw1 = np.dot(dw2,w2.T) * tanh(z2,deriv=True)
	dw0 = np.dot(dw1,w1.T) * tanh(z1,deriv=True)
#dw2=dumps(dw2.flatten().tolist())
#dw1=dumps(dw1.flatten().tolist())
#dw0=dumps(dw0.flatten().tolist())
#loss=dumps(loss.tolist())
	print ("w2:",dumps(dw2.tolist()))
	print ("w1:",dumps(dw1.tolist()))
	print ("w0:",dumps(dw0.tolist()))
	print ("loss:",loss)
	
	
			

