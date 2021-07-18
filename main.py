import flask
from app import app
from flask import Flask, request, redirect, jsonify
import pickle
model_heartdisease = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def main():
	return "Flask Api is initialized."

@app.route('/classify', methods=['GET'])
def classify():
	if flask.request.method == 'GET':
		Age= request.args.get('age')
		EstimatedSalary = request.args.get('salary')
		prediction = model_heartdisease.predict([[Age,EstimatedSalary ]])
		if prediction == 1:
			predicted_class = "It is chance to purchase the items from Ads from social media"
			result = 'We think that is {}.'.format(predicted_class)
			return result
		else:
			predicted_class = "There is not change to purchase from Ads from social media"
			result = 'We think that is {}.'.format(predicted_class)
			return result
	else:
		return "Select GET Method"

if __name__ == "__main__":
	app.run()