#pip install xgboost
import pandas as pd
from xgboost import XGBClassifier
from numpy import mean
from numpy import std
df_diabetes = pd.read_csv('diabetes.csv')
# We will scale the features using standardization
y = df_diabetes['Outcome']
X = df_diabetes.drop('Outcome', axis=1, inplace=False)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                             'Insulin','BMI','DiabetesPedigreeFunction','Age'])

from sklearn.model_selection import train_test_split,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# define the model
model = XGBClassifier(use_label_encoder=False)

# fit the model on dataset
model.fit(X_train, y_train)
# Accuracy
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print('Accuracy XBoost')
print(accuracy_score(y_test, y_pred))
# Bagging Classifier
from sklearn.ensemble import BaggingClassifier
# define the model
model_bagg = BaggingClassifier()
# fit the model on dataset
model_bagg.fit(X_train, y_train)

# Accuracy
y_pred_bagg = model.predict(X_test)
print('Accuracy Bagging Classifier')
print(accuracy_score(y_test, y_pred_bagg))
print('+++++++++++++++++++++++++++++++++++++++++++++++=')

#Stacking for Classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier
# get a list of base models
# get a stacking ensemble of models

def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('lr', LogisticRegression()))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('dt', DecisionTreeClassifier()))
    level0.append(('rf', RandomForestClassifier()))
    level0.append(('ada', AdaBoostClassifier()))
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

# get a list of models to evaluate
def get_models():
    models = dict()
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['dt'] = DecisionTreeClassifier()
    models['RF'] = RandomForestClassifier()
    models['ada'] = AdaBoostClassifier()
    models['stacking'] = get_stacking()
    return models

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f' % (name, mean(scores)))