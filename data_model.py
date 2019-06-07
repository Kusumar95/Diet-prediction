import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

balance_data = pd.read_csv('/home/mlh-admin/Desktop/input.csv',sep= ',')

X = balance_data.values[:, 1:-2]
Y = balance_data.values[:,-1]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

print(clf_gini.score(X_test,y_test))


clf = RandomForestClassifier()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


y_pred = clf.predict(X_test)
print(f1_score(y_test, y_pred, average='weighted'))
