import flask
from flask import Flask, request
import pickle
import pandas as pd
model_linear = pickle.load(open('modelRegression.pkl', 'rb'))
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    return "Flask Api is initialized."

@app.route('/predict', methods=['GET'])
def classify():
    if flask.request.method == 'GET':
        age = request.args.get('age')
        sex = request.args.get('sex')
        bmi = request.args.get('bmi')
        children = request.args.get('children')
        smoker = request.args.get('smoker')
        region = request.args.get('region')
        columns = [(age, sex, bmi, children, smoker, region)]
        labels = ['age','sex','bmi','children','smoker','region']
        predict = pd.DataFrame.from_records(columns, columns=labels)

        predict.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
        predict.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
        predict.replace({'region': {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)
        print(predict)
        age = predict.at[0,'age']
        sex = predict.at[0, 'sex']
        bmi = predict.at[0, 'bmi']
        children = predict.at[0, 'children']
        smoker = predict.at[0, 'smoker']
        region = predict.at[0, 'region']
        prediction = model_linear.predict([[age,sex,bmi,children,smoker,region]])
        result = 'We think that is {}.'.format(prediction)
        return result
    else:
        return "Select GET Method"
if __name__ == "__main__":
    app.run()