import csv
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Read data from files
data = np.array(pd.read_csv("../Dataset/imputed_train_median.csv"))
labels = np.array(pd.read_csv("../Dataset/train_labels.csv"))

#Shuffling the training set
data = np.column_stack((data,labels))
np.random.shuffle(data)

# 80% as training data and 20% as validation dataset
# n = int(0.8*(data.shape[0]))

# train_labels = data[:n,-1]
# train_data = data[:n,:-1]

# test_labels = data[n:,-1]
# test_data = data[n:,:-1]
# # print(train_labels[:35])
train_data, test_data, train_labels, test_labels = train_test_split(labels, y,stratify=y, test_size=0.20)


lda = LinearDiscriminantAnalysis(n_components=1000)
lda.fit(train_data, train_labels)
# X_train = lda.transform(train_data)

# # Prediction by LDA
y_prediction	=	lda.predict(test_data) # Predict the output for the test cases

print(classification_report(test_labels, y_prediction))