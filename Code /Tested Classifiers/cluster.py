import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as met
from sklearn.cluster import KMeans




plt.figure(figsize=(5, 5))

n_samples = 1500
random_state = 170
#X=np.loadtxt("1_Clustering/D31.csv",delimiter=',',skiprows=1)



class_labels=[]
for i in range(1,30):
	class_labels.append('class'+str(i))

X_test=np.loadtxt("../../newdataset/clean_test.csv",delimiter=',')
print(X_test.shape)
for i in [1,2,3,4,5]:

	X_train=np.loadtxt("../../newdataset/cross_val_dataset/clean_X_train_"+str(i)+".csv",delimiter=',')
	X_validate=np.loadtxt("../../newdataset/cross_val_dataset/clean_X_validate_"+str(i)+".csv",delimiter=',')
	Y_train=np.loadtxt("../../newdataset/cross_val_dataset/Ytrain_"+str(i)+".csv",delimiter=',')
	Y_validate=np.loadtxt("../../newdataset/cross_val_dataset/Yvalidate_"+str(i)+".csv",delimiter=',')
	print("Done")
	print(X_train.shape,Y_train.shape)

#for submission
	km=KMeans(n_clusters=29, random_state=random_state)
	y_pred = km.fit_predict(X_train)

	print(met.homogeneity_score(Y_train,y_pred))


	Y_out=np.zeros([X_test.shape[0],2])
	Y_out[:,0]=np.arange(X_test.shape[0])
	Y_out[:1]=km.predict(X_test)
	print(met.homogeneity_score(Y_train,y_pred))
	
	#pickle.dump(mlp,open("../../datasets/csv_file/model_"+str(i)+".pkl",'wb'))

	#print(mlp.score(X_validate,Y_validate))
	#print(met.classification_report(Y_validate,mlp.predict(X_validate), target_names=class_labels))

	Y_out[:,1]=mlp.predict(X_test)

	np.savetxt("../../datasets/csv_file/Y_out_clean"+str(i)+".csv", Y_out, fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')

