from flask import Flask, jsonify, render_template, request
from ML.deploy import DeployModel

dm = DeployModel('')
app = Flask(__name__)

@app.route('/_model_data')
def model_data():
	return jsonify(result=dm.get_data_and_model()[0])
	
@app.route('/_model_output')
def model_output():
	return jsonify(result=dm.get_data_and_model()[1])


@app.route('/')
def index():
	return render_template('index.html')
