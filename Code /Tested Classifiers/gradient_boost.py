import csv
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			fname.write("id,label\n")
			np.savetxt(fname,data,delimiter=',',newline='\n')

# Read data from files
# data = np.genfromtxt('../Dataset/train_mean_750_pca.csv', delimiter=',')
# data = np.genfromtxt('../Dataset/train_mean_1000_lda.csv', delimiter=',')
# labels = np.genfromtxt('../Dataset/train_labels.csv', delimiter=',')
# test = np.genfromtxt("../Dataset/test_mean_1000_lda.csv", delimiter=',')
test_data = np.genfromtxt("../Dataset/test_mean_500_pca.csv", delimiter=',')
# X = np.genfromtxt("../Dataset/X_mean_500_pca.csv", delimiter=',')
# labels = np.genfromtxt("../Dataset/train_labels.csv", delimiter=',')

# Read data from files
train_data = np.genfromtxt('../Dataset/temp/X_train_p.csv', delimiter=',')
train_labels = np.genfromtxt('../Dataset/temp/Y_train.csv', delimiter=',')
validation_data = np.genfromtxt("../Dataset/temp/X_validate_p.csv", delimiter=',')
validation_labels = np.genfromtxt("../Dataset/temp/Y_validate.csv", delimiter=',')

# print(data.shape)
# print(labels.shape)

# train_data, test_data, train_labels, test_labels = train_test_split(data, labels, stratify=labels, test_size=0.20, shuffle=True)


# Validation
print("\n\n\nRunning Validation..................................\n")

nn = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.05, max_depth=2, random_state=0, verbose=1)
nn.fit(train_data, train_labels)

y_prediction =nn.predict(validation_data)

item = np.column_stack((np.arange(len(y_prediction)), y_prediction))

csv_writer(item, "../Dataset/result_gb_valid3.csv","wb")

print(classification_report(validation_labels, y_prediction))

#Train
y_prediction =nn.predict(train_data)

item = np.column_stack((np.arange(len(y_prediction)), y_prediction))

csv_writer(item, "../Dataset/result_gb_train3.csv","wb")

# Test
print("\n\nRunning test........................")

y_prediction =nn.predict(test_data)
item = np.column_stack((np.arange(len(y_prediction)), y_prediction))

csv_writer(item, "../Dataset/result_gb_test3.csv","wb")

print(nn.score(validation_data, validation_labels))

print("\n\nDone :)")
