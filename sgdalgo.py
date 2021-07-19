import pandas as pd
import pickle
from sklearn.pipeline import make_pipeline

import pandas as pd
import pickle

data = pd.read_csv(r'Iris.csv')
print(data)

from sklearn.model_selection import train_test_split
X = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]  # Features
y = data['Species']  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(loss="hinge",penalty="l2", max_iter=1000, random_state=None, learning_rate='optimal')
classifier.fit(X, y)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(accuracy_score(y_test, y_pred))

# clf = make_pipeline(SGDClassifier(max_iter=1000, tol=1e-3))
# clf.fit(X, y)
#
# # Predicting the Test set results
# y_pred = clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))

with open('modelIRIS.pkl', 'wb') as file:
    pickle.dump(classifier, file)
