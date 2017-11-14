from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
import sklearn.metrics as met
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

class_labels=[]
for i in range(1,30):
	class_labels.append('class'+str(i))

X_test=np.loadtxt("../../newdataset/test.csv",delimiter=',')[:,1:]
Y_out=np.zeros([X_test.shape[0],2])
Y_out[:,0]=np.arange(X_test.shape[0])


for i in range(1,6):

	X_train=np.loadtxt("../../newdataset/cross_val_dataset/X_train_"+str(i)+".csv",delimiter=',')[:,1:]
	X_validate=np.loadtxt("../../newdataset/cross_val_dataset/X_validate_"+str(i)+".csv",delimiter=',')[:,1:]
	Y_train=np.loadtxt("../../newdataset/cross_val_dataset/Ytrain_"+str(i)+".csv",delimiter=',')
	Y_validate=np.loadtxt("../../newdataset/cross_val_dataset/Yvalidate_"+str(i)+".csv",delimiter=',')
	print("done loading")
	print("training.....")
	 
	clf1 = LogisticRegression(random_state=1)
	clf2 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
	max_depth=30, max_features='auto', max_leaf_nodes=None,
	min_samples_leaf=1, min_samples_split=2,
	min_weight_fraction_leaf=0.0, n_estimators=1800, n_jobs=1,random_state=1)
	clf3 = GaussianNB()
	#clf4=KNeighborsClassifier(n_neighbors=20)
	
	clf4=MLPClassifier(hidden_layer_sizes=(800,800,100), activation='relu', solver='adam', 
	alpha=0.01, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.005, 
	power_t=0.5, max_iter=10500, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
	warm_start=False, momentum=0.5, nesterovs_momentum=True, epsilon=1e-08)
	
 	#eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),('nn',clf4)], voting='hard')

	bagging = BaggingClassifier(VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),('knn',clf4)], voting='hard',weights=None),max_samples=0.4, max_features=0.3)

	bagging.fit(X_train,Y_train)

	Y_val=bagging.predict(X_validate)
	Y_out[:,1]=bagging.predict(X_test)

	print(met.classification_report(Y_validate,Y_val, target_names=class_labels))
	np.savetxt("../../newdataset/Y_out_"+str(i)+".csv", Y_out, fmt='%d',delimiter=',', newline='\n', header='id,label',comments='')
