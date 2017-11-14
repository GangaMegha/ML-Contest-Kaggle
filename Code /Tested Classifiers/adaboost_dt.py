import csv
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold

def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			fname.write("id,label\n")
			np.savetxt(fname,data,delimiter=',',newline='\n')

# Read data from files
#LDA
# train_data = np.genfromtxt('../Dataset/train_mean_28_lda.csv', delimiter=',')
# train_labels = np.genfromtxt('../Dataset/train_labels_mean_28_lda.csv', delimiter=',')
# validation_data = np.genfromtxt("../Dataset/validation_mean_28_lda.csv", delimiter=',')
# validation_labels = np.genfromtxt("../Dataset/validation_labels_mean_28_lda.csv", delimiter=',')
# test_data = np.genfromtxt("../Dataset/test_mean_28_lda.csv", delimiter=',')
# X = np.genfromtxt("../Dataset/X_mean_28_lda.csv", delimiter=',')

train_data = np.genfromtxt('../Dataset/temp/X_train_p.csv', delimiter=',')
train_labels = np.genfromtxt('../Dataset/temp/Y_train.csv', delimiter=',')
validation_data = np.genfromtxt("../Dataset/temp/X_validate_p.csv", delimiter=',')
validation_labels = np.genfromtxt("../Dataset/temp/Y_validate.csv", delimiter=',')
test_data = np.genfromtxt("../Dataset/test_mean_500_pca.csv", delimiter=',')
# X = np.genfromtxt("../Dataset/X_mean_28_lda.csv", delimiter=',')

# #PCA
# train_data = np.genfromtxt('../Dataset/train_mean_500_pca.csv', delimiter=',')
# train_labels = np.genfromtxt('../Dataset/train_labels_mean_500_pca.csv', delimiter=',')
# validation_data = np.genfromtxt("../Dataset/validation_mean_500_pca.csv", delimiter=',')
# validation_labels = np.genfromtxt("../Dataset/validation_labels_mean_500_pca.csv", delimiter=',')
# test_data = np.genfromtxt("../Dataset/test_mean_500_pca.csv", delimiter=',')
# X = np.genfromtxt("../Dataset/X_mean_500_pca.csv", delimiter=',')



# Validation
print("\n\n\nRunning Validation..................................\n")

bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7), n_estimators=1000, learning_rate=0.1)
bdt_real.fit(train_data, train_labels)

y_prediction = bdt_real.predict(validation_data)

item = np.column_stack((np.arange(len(y_prediction)), y_prediction))

csv_writer(item, "../Dataset/result_ab_valid.csv","wb")

print(classification_report(validation_labels, y_prediction))

#Train
y_prediction = bdt_real.predict(train_data)

item = np.column_stack((np.arange(len(y_prediction)), y_prediction))

csv_writer(item, "../Dataset/result_ab_train.csv","wb")

# Test
print("\n\nRunning test........................")
# rf = RandomForestClassifier(n_estimators=3000, max_depth=25, min_samples_split=4, min_samples_leaf=2)
# rf.fit(X, labels)
# rf.fit(train_data,train_labels)

y_prediction = bdt_real.predict(test_data)
item = np.column_stack((np.arange(len(y_prediction)), y_prediction))

csv_writer(item, "../Dataset/result_ab_test.csv","wb")

print("\n\nDone :)")

# # Test
# print("\n\nRunning test........................")

# bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20), n_estimators=2000, learning_rate=1)
# bdt_real.fit(train_data, train_labels)
# y_prediction = bdt_real.predict(test_data)

# item = np.column_stack((np.arange(len(y_prediction)), y_prediction))

# csv_writer(item, "../Dataset/result_adaboost_dt.csv","wb")

# print("\n\nDone :)")