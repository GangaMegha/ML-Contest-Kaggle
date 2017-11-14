import numpy as np 
import pickle

test_data = np.loadtxt('../Dataset/test.csv', delimiter=',')

for i in range(5):
	n=i+1
	model=pickle.load(open("../Models/rf_full{}.model".format(n),'rb'))
	y_out=model.predict(test_data)
	Y = np.zeros((y_out.shape[0],2))
	Y[:,0] = np.arange(y_out.shape[0])
	Y[:,1] = y_out
	np.savetxt("../Results/result_test_{}.csv".format(i+1), Y, fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')
