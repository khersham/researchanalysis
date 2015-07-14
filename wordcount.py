from pandas import read_csv
import numpy as np

csv_reader = read_csv("/home/lim/Documents/mapreduce/part-r-00000", "\t",header=0,names=['Word','Count'])
csv_reader2 = read_csv("/home/lim/Documents/mapreduce/part-r-00001", "\t",header=0,names=['Word','Count'])

datap1 = csv_reader.sort('Count',ascending=False)
datap2 = csv_reader2.sort('Count',ascending=False)

datap1[:10].plot(kind='bar')

temp = datap1['Word'][:200].values
np.savetxt("/home/lim/Documents/mapreduce/common.txt",temp,delimiter=",",fmt='%s')

temp = datap2['Word'][:200].values
np.savetxt("/home/lim/Documents/mapreduce/common1.txt",temp,delimiter=",",fmt='%s')