import numpy as np 

data = np.genfromtxt('data1_10_kmeans.csv', delimiter=',',skip_header=1, type=None)
train_data = np.genfromtxt('../Dataset/train_cv_p1.csv', delimiter=',')
validation_data = np.genfromtxt("../Dataset/valid_cv_p1.csv", delimiter=',')
test_data = np.genfromtxt("../Dataset/test_cv_p1.csv", delimiter=',')

cluster = data[:,-1]
C = np.zeros((cluster.shape[0],10))
print(data[:10,-1])
for i in range(cluster.shape[0]):
	print(cluster[i])
	print(data[i,-1])
	k = int(cluster[i,7:])
	C[i][k] = 1

data = data[:,:-1]
data = np.column_stack((data,C))

train_data = data[:train_data.shape[0]]
validation_data = data[train_data.shape[0]:train_data.shape[0]+validation_data.shape[0]]
test_data = data[train_data.shape[0]+validation_data.shape[0]:]

np.savetxt("../Dataset/TRAIN.csv".format(i+1), train_data, fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')
np.savetxt("../Dataset/VALIDATE.csv".format(i+1), validation_data, fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')
np.savetxt("../Dataset/TEST.csv".format(i+1), test_data, fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')

