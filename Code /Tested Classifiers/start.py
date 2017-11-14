import numpy as np 
from sklearn.preprocessing import Imputer
#from sklearn import linear_model
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

impute=Imputer(missing_values='NaN',strategy='median',axis=0)
train_data=impute.fit_transform(train_data[:,:-1])
test=impute.transform(test)



for i in [1,2,3,4,5]:

	X_train, X_validate, Y_train, Y_validate = train_test_split(train_data, train_labels,stratify=train_labels, test_size=0.15,random_state=seed[i-1])
	csv_writer(X_train,"../../datasets/csv_file/cross_val_data_set/X_train_"+str(i)+".csv","wb")
	csv_writer(X_validate,"../../datasets/csv_file/cross_val_data_set/X_validate_"+str(i)+".csv","wb")
	csv_writer(Y_train,"../../datasets/csv_file/cross_val_data_set/Y_train_"+str(i)+".csv","wb")
	csv_writer(Y_validate,"../../datasets/csv_file/cross_val_data_set/Y_validate_"+str(i)+".csv","wb")
	print("Done")


#csv_writer(train_data[:,1:-1],"imputed_train_median.csv","wb")
#csv_writer(test[:,1:],"imputed_test_median.csv","wb")



