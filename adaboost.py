'''
Install Libraraies
1. pip install scikit-learn
2. pip install pandas
'''
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\Ashwini\Desktop\Academic\dataset\addsdataset.csv')
print(dataset)
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting to Training set
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

print("Result after applying the Parameter tuning")

"""Using Random forest as a base estimator"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
RF = RandomForestClassifier(max_depth=2, random_state=0)
adb = AdaBoostClassifier(n_estimators=100,base_estimator = RF, learning_rate= 0.01, random_state = 42)
adb.fit(X_train, y_train)
# Predicting the Test set results
y_pred = adb.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Save the model, we will used in flask

with open('model.pkl','wb') as file:
    pickle.dump(classifier, file)

with open('modelNew.pkl','wb') as file:
    pickle.dump(adb, file)

print("Done")


