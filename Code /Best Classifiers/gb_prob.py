#GRADIENT BOOSTING : Using Probabilities

import csv
import numpy as np
import pandas as pd
from scipy.stats import mode
import pickle

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold


def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			fname.write("id,label\n")
			np.savetxt(fname,data,delimiter=',',newline='\n')

with open("../Results/gb_prob_results.txt", "w") as text_file:

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

		gb = GradientBoostingClassifier(n_estimators=1500, learning_rate=0.1, max_depth=2, random_state=0, verbose=1)
		gb.fit(train_data,train_labels)

		# Save model
		pickle.dump(gb, open("../Models/gb_prob{}.model".format(i+1), 'wb'))

		if(i!=0):
			y_prediction_valid = y_prediction_valid + gb.predict_proba(validation_data)
		else:
			y_prediction_valid = gb.predict_proba(validation_data)

		y_prediction = gb.predict(validation_data)

		text_file.write("\nCross-Validation {}".format(i+1))
		text_file.write(classification_report(validation_labels, y_prediction))
		text_file.write("\n\n\n")

		#Train
		if(i!=0):
			y_prediction_train = y_prediction_train + gb.predict_proba(train_data)
		else:
			y_prediction_train = gb.predict_proba(train_data)

		# Test
		print("\n\nRunning test........................")

		if(i!=0):
			y_prediction_test = y_prediction_test + gb.predict_proba(test_data)
		else:
			y_prediction_test = gb.predict_proba(test_data)

t1 = np.argmax(y_prediction_valid,axis=1).reshape(y_prediction_valid.shape[0])
	
t2 = np.argmax(y_prediction_train,axis=1).reshape(y_prediction_train.shape[0])

t3 = np.argmax(y_prediction_test,axis=1).reshape(y_prediction_test.shape[0])

t1 = np.column_stack((np.arange(len(t1)), t1))
t2 = np.column_stack((np.arange(len(t2)), t2))
t3 = np.column_stack((np.arange(len(t3)), t3))

csv_writer(t1, "../Results/gb_prob_valid.csv","wb")
csv_writer(t2, "../Results/gb_prob_train.csv","wb")
csv_writer(t3, "../Results/gb_prob_test.csv","wb")

print("\n\nDone :)")