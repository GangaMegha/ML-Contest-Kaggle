import numpy as np 
from sklearn.preprocessing import Imputer,scale
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			np.savetxt(fname,data,delimiter=',',newline='\n')


train_data=np.genfromtxt('../../datasets/csv_file/train.csv', delimiter=',',missing_values="NaN",skip_header=1)
test=np.genfromtxt('../../datasets/csv_file/test.csv', delimiter=',',missing_values="NaN",skip_header=1)

seed=[97,72,4,56,34]

print(train_data.shape)
print(test.shape)

train_labels=train_data[:,-1]
u,indices=np.unique(train_labels,return_index=True)

print(train_data.shape[0])
list(indices).append(train_data.shape[0])

print(indices)
lr=LinearRegression.fit()

impute=Imputer(missing_values='NaN',strategy='median',axis=0)
for i in range(28):
	print(i,indices[i],indices[i+1])
	train_data[indices[i]:indices[i+1],:-1]=impute.fit_transform(train_data[indices[i]:indices[i+1],:-1])

#lr=LinearRegression(normalise=False)
#lr.fit(train_data[501])

train_data[indices[-1]:,:-1]=scale(impute.fit_transform(train_data[indices[-1]:,:-1]))

impute.fit(train_data[:,:-1])
test=scale(impute.transform(test))

csv_writer(train_data[:,501:-1],"../../newdataset/cctrain.csv","wb")
csv_writer(test[:,501:],"../../newdataset/clean_test.csv","wb")
	

for i in [1,2,3,4,5]:

	X_train, X_validate, Y_train, Y_validate = train_test_split(train_data[:,501:-1], train_labels,stratify=train_labels, test_size=0.15,random_state=seed[i-1])
	csv_writer(X_train,"../../newdataset/cross_val_dataset/clean_X_train_"+str(i)+".csv","wb")
	csv_writer(X_validate,"../../newdataset/cross_val_dataset/clean_X_validate_"+str(i)+".csv","wb")
	csv_writer(Y_train,"../../newdataset/cross_val_dataset/Ytrain_"+str(i)+".csv","wb")
	csv_writer(Y_validate,"../../newdataset/cross_val_dataset/Yvalidate_"+str(i)+".csv","wb")
	print("Done")



