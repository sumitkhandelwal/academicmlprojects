import pandas as pd
import numpy as np

df_diabetes = pd.read_csv('diabetes.csv')

# We will scale the features using standardization

y = df_diabetes['Outcome']
X = df_diabetes.drop('Outcome', axis=1, inplace=False)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                             'Insulin','BMI','DiabetesPedigreeFunction','Age'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# Logistic Regression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
y_pred_logreg = logreg.predict(X_test)
print('Accuracy Logistic Regression')
print(accuracy_score(y_test, y_pred_logreg))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier()
dectree.fit(X_train,y_train)
y_pred_dectree = dectree.predict(X_test)
print('Accuracy Decision Tree Classifier')
print(accuracy_score(y_test, y_pred_dectree))
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print('Accuracy KN Neighbors Classifier')
print(accuracy_score(y_test, y_pred_knn))
# Random Forest
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators=1000, random_state=0)
ranfor.fit(X_train, y_train)
y_pred_ranfor = ranfor.predict(X_test)
print('Accuracy Random Forest')
print(accuracy_score(y_test, y_pred_ranfor))
# Ada Boost Classifier
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=1000)
abc.fit(X_train, y_train)
y_pred_abc = abc.predict(X_test)
print('Accuracy Ada Boost Classifier')
print(accuracy_score(y_test, y_pred_ranfor))

# Voting Classifier without weights

from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(estimators=[('logreg',logreg),('dectree',dectree),('ranfor',ranfor),('knn',knn),('abc',abc)],
                      voting='soft')
vc.fit(X_train, y_train)
y_pred_vc = vc.predict(X_test)
print('Accuracy Voting Classifier')
print(accuracy_score(y_test, y_pred_vc))
#-----------------------------------------------------#
# get a list of base models
def get_models():
    models = list()
    models.append(('lr', LogisticRegression()))
    models.append(('dectree', DecisionTreeClassifier()))
    models.append(('ranfor', RandomForestClassifier()))
    models.append(('knn', KNeighborsClassifier()))
    models.append(('abc', AdaBoostClassifier()))
    return models

# evaluate each base model # fit and evaluate the models
# evaluate each base model
def evaluate_models(models, X_train, X_test, y_train, y_test):
    # fit and evaluate the models
    scores = list()
    for name, model in models:
        # fit the model
        model.fit(X_train, y_train)
        # evaluate the model
        yhat = model.predict(X_test)
        acc = accuracy_score(y_test, yhat)
        # store the performance
        scores.append(acc)
    # report model performance
    return scores

# create the base models
models = get_models()
# fit and evaluate each model
scores = evaluate_models(models, X_train, X_test, y_train, y_test)
print(scores)

# Voting Classifier with weights

vc1 = VotingClassifier(estimators=models,
                       voting='soft', weights=scores)
vc1.fit(X_train, y_train)
y_pred_vc1 = vc1.predict(X_test)
print('Accuracy Voting Classifier (SOFT)')
print(accuracy_score(y_test, y_pred_vc1))

# Voting Classifier with weights
vc1 = VotingClassifier(estimators=models,
                       voting='hard', weights=scores)
vc1.fit(X_train, y_train)
y_pred_vc1 = vc1.predict(X_test)
print('Accuracy Voting Classifier(HARD)')
print(accuracy_score(y_test, y_pred_vc1))