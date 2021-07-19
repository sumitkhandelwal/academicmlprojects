from os import path
import pickle
from flask import Flask, request, redirect, jsonify
app = Flask(__name__)
model_adaboost = pickle.load(open('modelIRIS.pkl', 'rb'))
@app.route('/file-upload', methods=['POST'])
def upload_file():
	content = request.get_json()
	print(content)
	SepalLengthCm = content['sl']
	SepalWidthCm = content['sw']
	PetalLengthCm = content['pl']
	PetalWidthCm = content['pw']
	prediction = model_adaboost.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
	return 'Class of species is ' + str(prediction)

@app.route('/file', methods=['GET'])
def list():
	fileName = request.args.get('file')
	if(path.exists('upload/'+fileName) == True):
		return 'Your file is present at location upload/'+fileName

if __name__ == "__main__":
    app.run()

''' 
Post Body Part
{
	"sl":"5.6",
	"sw":3.4,
	"pl":3.4,
	"pw":0.2
}
'''