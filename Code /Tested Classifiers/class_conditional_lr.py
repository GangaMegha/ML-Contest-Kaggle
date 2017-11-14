import csv
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold


def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			# fname.write("id,label\n")
			np.savetxt(fname,data,delimiter=',',newline='\n')

train_data=np.genfromtxt('../Dataset/train.csv', delimiter=',',missing_values="NaN",skip_header=1)
test=np.genfromtxt('../Dataset/test.csv', delimiter=',',missing_values="NaN",skip_header=1)
labels = np.genfromtxt("../Dataset/train_labels.csv", delimiter=',')

# train_data, test_data, train_labels, test_labels = train_test_split(data, labels, stratify=labels, test_size=0.20, shuffle=True)
x = train_data[:,502:-1].astype("float")
y = train_data[:,1:502]

y_pred = np.zeros_like(y)

for i in range(500):

	temp = y[:,i]
	# print(temp)
	index1 = np.where(temp<9999)
	index2 = np.argwhere(np.isnan(temp))
	print(index2.shape)
	index2 = index2.reshape(index2.shape[0])

	# print(index1)
	print(index2)

	x_train = x[index1]
	y_train = temp[index1].astype("float")
	print(x_train.shape)

	x_test = x[index2].astype("float")

	# Create linear regression object
	regr = linear_model.LinearRegression()


	regr.fit(x_train, y_train)

	# Make predictions using the testing set
	pred = regr.predict(x_test)

	temp2 = np.zeros(temp.shape[0])
	temp2[index1] = temp[index1].astype("float")
	temp2[index2] = pred

	y_pred[:,i] = temp2

Train = np.zeros((train_data.shape[0], train_data.shape[1]-2))
Train[:,:501] = y_pred
Train[:,501:] = x.astype("float")
csv_writer(Train, "../Dataset/LR_Train.csv", "wb")


x = test[:,502:].astype("float")
y = test[:,1:502]

y_pred = np.zeros_like(y)

for i in range(500):

	temp = y[:,i]
	# print(temp)
	index1 = np.where(temp<9999)
	index2 = np.argwhere(np.isnan(temp))
	print(index2.shape)
	index2 = index2.reshape(index2.shape[0])

	# print(index1)
	print(index2)

	x_train = x[index1]
	y_train = temp[index1].astype("float")
	print(x_train.shape)

	x_test = x[index2].astype("float")

	# Create linear regression object
	regr = linear_model.LinearRegression()


	regr.fit(x_train, y_train)

	# Make predictions using the testing set
	pred = regr.predict(x_test)

	temp2 = np.zeros(temp.shape[0])
	temp2[index1] = temp[index1].astype("float")
	temp2[index2] = pred

	y_pred[:,i] = temp2

Test = np.zeros((test.shape[0], test.shape[1]-1))
Test[:,:501] = y_pred
Test[:,501:] = x.astype("float")
csv_writer(Test, "../Dataset/LR_Test.csv", "wb")