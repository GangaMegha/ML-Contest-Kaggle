# XGBOOST

from scipy.stats import mode

import csv
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier


def csv_writer(data,filename,mode):
	np.savetxt(filename, data, fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')

		# with open(filename,mode) as fname:
		# 	fname.write("id,label\n")
		# 	np.savetxt(fname,data,delimiter=',',newline='\n')



with open("../Results/ensemble_results.txt", "w") as text_file:
	for i in range(5):

		# Read data from files
		#LDA
		# train_data = np.genfromtxt('../Dataset/train_cv_l{}.csv'.format(i+1), delimiter=',')
		# train_labels = np.genfromtxt('../Dataset/train_labels{}.csv'.format(i+1), delimiter=',')
		# validation_data = np.genfromtxt("../Dataset/valid_cv_l{}.csv".format(i+1), delimiter=',')
		# validation_labels = np.genfromtxt("../Dataset/valid_labels{}.csv".format(i+1), delimiter=',')
		# test_data = np.genfromtxt("../Dataset/test_cv_l.csv".format(i+1), delimiter=',')

		#PCA
		train_data = np.genfromtxt('../Dataset/train_cv_p{}.csv'.format(i+1), delimiter=',')
		train_labels = np.genfromtxt('../Dataset/train_labels_{}.csv'.format(i+1), delimiter=',')
		validation_data = np.genfromtxt("../Dataset/valid_cv_p{}.csv".format(i+1), delimiter=',')
		validation_labels = np.genfromtxt("../Dataset/valid_labels_{}.csv".format(i+1), delimiter=',')
		test_data = np.genfromtxt("../Dataset/test_cv_p{}.csv".format(i+1), delimiter=',')


		# Validation
		print("\n\n\nRunning Validation..................................\n")

		rf=pickle.load(open("../Models/rf_full{}.model".format(i+1),'rb'))
		gb=pickle.load(open("../Models/gb{}.model".format(i+1),'rb'))
		xgb=pickle.load(open("../Models/xgbc{}_1.model".format(i+1),'rb'))


		obj = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)],voting='soft')

		obj.fit(train_data,train_labels)

		# Save model
		pickle.dump(obj, open("../Models/ensemble{}.model".format(i+1), 'wb'))

		y_prediction = obj.predict(validation_data)

		text_file.write("\nCross-Validation {}".format(i+1))
		text_file.write(classification_report(validation_labels, y_prediction))
		text_file.write("\n\n\n")

		print("\n\nRunning test........................")

		y_prediction = obj.predict(test_data)

		if(i!=0):
			item3 = np.column_stack((item3, y_prediction))
		else:
			item3 = y_prediction
		print(item3.shape)

t3 = mode(np.array(item3), axis=1)[0].reshape(item3.shape[0])

t3 = np.column_stack((np.arange(len(t3)), t3))

csv_writer(t3, "../Results/ensemble_test.csv","wb")

print("\n\nDone :)")