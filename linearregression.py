import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("insurance.csv")
#print(dataset.shape)

# convert the categorical values to int
#convert sex column to male=0,female=1
dataset.replace({'sex':{'male':0, 'female':1}},inplace=True)

dataset.replace({'smoker':{'yes':0, 'no':1}},inplace=True)
dataset.replace({'region':{'southwest':0, 'southeast':1,'northwest':2,'northeast':3}},inplace=True)
#print(dataset.head())
#remove the charges column as Target column Y and dependent columns as X
X = dataset.drop(columns = 'charges')
Y = dataset['charges']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.2, random_state = 3)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)

#train data prediction
prediction_train = model.predict(X_train)

from sklearn.metrics import r2_score
optimization = r2_score(Y_train, prediction_train )
print(optimization)

# Predicting the Test set results
prediction_test  = model.predict(X_test)
optimization = r2_score(Y_test, prediction_test)
print(optimization)
#############
import pickle
with open('modelRegression.pkl','wb') as file:
    pickle.dump(model, file)