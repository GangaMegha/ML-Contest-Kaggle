import numpy as np 
import matplotlib.pyplot as plt 

def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			np.savetxt(fname,data,delimiter=',',newline='\n')


X=np.genfromtxt('../dataset/train.csv', delimiter=',',missing_values="NaN",skip_header=1)
Y=np.genfromtxt('../dataset/test.csv', delimiter=',',missing_values="NaN",skip_header=1)
print(X.shape)
print(Y.shape)
#X_validate=np.loadtxt('../../datasets/csv_file/X_validate_p.csv', delimiter=',')
#X_test=np.loadtxt('../../datasets/csv_file/X_test_p.csv', delimiter=',')
'''

row_count=np.count_nonzero(np.isnan(X),axis=1)
col_count=np.count_nonzero(np.isnan(X),axis=0)

print(row_count.size)
print(col_count.size)
print(row_count)
print(col_count)

plt.plot(row_count)
plt.show()
plt.plot(col_count)
plt.show()
'''
X_clean=X[:,501:-1]
Y_clean=Y[:,501:]
print(X_clean.shape)
print(Y_clean.shape)

row_count=np.count_nonzero(np.isnan(X_clean),axis=1)
col_count=np.count_nonzero(np.isnan(X_clean),axis=0)
plt.plot(row_count)
plt.show()
plt.plot(col_count)
plt.show()

csv_writer(X_clean,'../dataset/clean_train.csv',"wb")
csv_writer(Y_clean,'../dataset/clean_test.csv',"wb")