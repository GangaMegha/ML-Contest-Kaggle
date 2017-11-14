import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import sklearn.metrics as met


class_labels=[]
for i in range(1,30):
	class_labels.append('class'+str(i))

#for pca
'''
#for lda
X_train=np.loadtxt('../../datasets/csv_file/X_train.csv', delimiter=',')
X_validate=np.loadtxt('../../datasets/csv_file/X_validate.csv', delimiter=',')
X_test=np.loadtxt('../../datasets/csv_file/X_test.csv', delimiter=',')
Y_train=np.loadtxt('../../datasets/csv_file/Y_train.csv', delimiter=',') 
Y_validate=np.loadtxt('../../datasets/csv_file/Y_validate.csv', delimiter=',')
'''


train_data=preprocessing.scale(np.loadtxt("../../datasets/csv_file/X_train_p_80.csv", delimiter=','))
valid_data=preprocessing.scale(np.loadtxt("../../datasets/csv_file/X_validate_p_80.csv", delimiter=','))
test_data=preprocessing.scale(np.loadtxt("../../datasets/csv_file/X_test_p_80.csv", delimiter=','))

train_labels=np.loadtxt('../../datasets/csv_file/Y_train.csv', delimiter=',')
valid_labels=preprocessing.scale(np.loadtxt("../../datasets/csv_file/Y_validate.csv", delimiter=','))
print("Done pre")

Y_out=np.zeros([test_data.shape[0],2])
Y_out[:,0]=np.arange(test_data.shape[0])#test_labels=np.loadtxt('../DS2/data_students/test_labels.csv', delimiter=',')

#c_set=[1e-4,1e-3,1e-2,1e-1,1,1e1,1e2]
c_set=[2**-8,2**-7,2**-6,2**-5,2**-4,2**-3,2**-2,2**-1,1,2**1,2**2]
coeff0_set=[1e-3,1e-2,1e-1,1,1e1]
gamma_set=[1e-4,1e-3,1e-2,1e-1,1,10,100]
deg=[1,2,3,4,5]

best_model_file="../../datasets/csv_file/results/model_svm_rbf"
max_acc=0
best_so_far=None
#kernels=['linear','rbf','poly','sigmoid']
i=0
for c in c_set:
	for gam in gamma_set:
		svc=SVC(C=c, kernel='rbf',gamma=gam, shrinking=True, 
			probability=False, tol=0.001, cache_size=128, class_weight=None, verbose=False, 
			max_iter=-1, decision_function_shape='ovr', random_state=7)

		score=cross_val_score(svc, train_data, y=train_labels, cv=10)
		print("C:"+str(c)+" gamma:"+str(gam))
		pickle.dump(svc,open(best_model_file+'_'+str(i)+'.pkl','wb'))
		print(np.mean(score))
		if max_acc<np.mean(score):
			max_acc=np.mean(score)
			best_so_far=str(best_model_file+'_'+str(i)+'.pkl')
			print(max_acc)
		i+=1






model=pickle.load(open(kernel+'.pkl','rb'))
Y_out[:,1]=model.predict(test_data)
valid_out=model.predict(valid_data)
print("Accuracy:"+str(met.accuracy_score(valid_labels,valid_out)))
print(met.confusion_matrix(valid_labels,valid_out))
print(met.classification_report(valid_labels,valid_out, target_names=class_labels))
np.savetxt("../../datasets/csv_file/results/Y_out_svm.csv", Y_out[:,1], fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')


'''
#met.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2)[source]¶
pickle.dump(mlp,open("../../datasets/csv_file/model_6.pkl",'wb'))

#print(mlp.score(X_validate,Y_validate))

Y_out[:,1]=mlp.predict(X_test)
print(met.classification_report(Y_validate, mlp.predict(X_validate), target_names=class_labels))

np.savetxt("../../datasets/csv_file/Y_out_6.csv", Y_out, fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')
'''