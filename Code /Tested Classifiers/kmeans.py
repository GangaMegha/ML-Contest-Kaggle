import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics


def csv_writer(data,filename,mode):
	np.savetxt(filename,data,delimiter=',',newline='\n')

data = np.genfromtxt('../Dataset/imputed_train_median.csv', delimiter=',')
test_data = np.genfromtxt('../Dataset/imputed_test_median.csv', delimiter=',')

kmeans = KMeans(n_clusters=29).fit(data)

train = kmeans.predict(data)
test = kmeans.predict(test_data)

data = np.column_stack((data,train)).astype("float")
test_data = np.column_stack((test_data,test)).astype("float")

csv_writer(data, "../Dataset/train_cluster.csv","wb")
csv_writer(test_data, "../Dataset/test_cluster.csv","wb")
 