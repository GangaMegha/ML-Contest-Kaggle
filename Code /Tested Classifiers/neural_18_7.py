from sklearn.neural_network import MLPClassifier
import numpy as np
import sklearn.metrics as met
import pickle


class_labels=[]
for i in range(1,30):
	class_labels.append('class'+str(i))

#for pca
X_test=np.loadtxt('../../datasets/csv_file/X_test.csv', delimiter=',')

Y_out=np.zeros([X_test.shape[0],2])
Y_out[:,0]=np.arange(X_test.shape[0])

for i in range(1,6):
	X_train=np.loadtxt('../../datasets/csv_file/cross_val_data_set/X_train_'+str(i)+'.csv', delimiter=',')
	X_validate=np.loadtxt('../../datasets/csv_file/cross_val_data_set/X_validate_'+str(i)+'.csv', delimiter=',')
	Y_train=np.loadtxt('../../datasets/csv_file/cross_val_data_set/Y_train_'+str(i)+'.csv', delimiter=',') 
	Y_validate=np.loadtxt('../../datasets/csv_file/cross_val_data_set/Y_validate_'+str(i)+'.csv', delimiter=',')
	
	mlp=MLPClassifier(hidden_layer_sizes=(1000,800,100), activation='relu', solver='adam', 
		alpha=0.01, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.005, 
		power_t=0.5, max_iter=10500, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
		warm_start=False, momentum=0.5, nesterovs_momentum=True, epsilon=1e-08)

	mlp.fit(X_train,Y_train)
	pickle.dump(mlp,open("../../datasets/csv_file/cross_out/model_"+str(i)+".pkl",'wb'))

	print(mlp.score(X_validate,Y_validate))

	#Y_out[:,1]=mlp.predict(X_test)
	print(met.classification_report(Y_validate, mlp.predict(X_validate)))#target_names=class_labels))

	#np.savetxt("../../datasets/csv_file/cross_out/Y_out_"+str(i)+".csv", Y_out, fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')






