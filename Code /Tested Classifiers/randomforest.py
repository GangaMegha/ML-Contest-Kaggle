import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier


X=np.loadtxt('../../datasets/csv_file/imputed_train_median.csv', delimiter=',')
#X_validate=np.loadtxt('../../datasets/csv_file/X_validate_p.csv', delimiter=',')
#X_test=np.loadtxt('../../datasets/csv_file/X_test_p.csv', delimiter=',')
Y=np.loadtxt('../../datasets/csv_file/train_labels.csv', delimiter=',') 
#Y_validate=np.loadtxt('../../datasets/csv_file/Y_validate.csv', delimiter=',')


forest = ExtraTreesClassifier(n_estimators=12500,
                              random_state=0)

forest.fit(X, Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()