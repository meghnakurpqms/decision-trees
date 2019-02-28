import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
#import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.metrics import confusion_matrix
import pydotplus
#matplotlib inline
col_head=['y','x1','x2','x3','x4','x5']
Mtrn = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
Mtst = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
print(Mtrn.shape)
#print(Mtrn.head())
clf = DecisionTreeClassifier()
ytrn = Mtrn[:, 0]
Xtrn = Mtrn[:, 1:]
ytst = Mtst[:, 0]
Xtst = Mtst[:, 1:]
clf = clf.fit(Xtrn,ytrn)
y_pred = clf.predict(Xtst)
print("Accuracy:",metrics.accuracy_score(ytst, y_pred))
matrix=confusion_matrix(ytst, y_pred)
print(matrix)
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = col_head,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('mydecision.png')

Image(graph.create_png())
