import numpy as np
import pickle
import os

file_dir = "/home/shivam/Documents/HadoopStreaming/"

w0 = np.random.uniform(low=-np.sqrt(6./(54675+100)), high=np.sqrt(6./(54675+100)),size= (54675,100))
w1 = np.random.uniform(low=-np.sqrt(6./(100+50)), high=np.sqrt(6./(100+50)),size=(100,50))
w2 = np.random.uniform(low=-np.sqrt(6./(50+1)), high=np.sqrt(6./(50+1)),size=(50,1))
b0 = np.random.normal(0., 0.25, (1, 100))
b1 = np.random.normal(0., 0.25, (1, 50))
b2 = np.random.normal(0., 0.25, (1, 1))

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

itr = 2

for i in range(0, itr):
	os.system("hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.6.0.jar "+\
		"-file "+file_dir+"nn-mapper.py "+\
		"-mapper "+file_dir+"nn-mapper.py "+\
		"-file "+file_dir+"nn-reducer.py "+\
		"-reducer "+file_dir+"nn-reducer.py "+\
		"-input /mrjob/sdata11.csv -output /mrjob/can"+str(i)+" > "+file_dir+"logs")