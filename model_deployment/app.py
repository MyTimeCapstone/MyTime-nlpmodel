from ast import Str
import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import json

app = Flask(__name__)
#load model
model = pickle.load(open('model_deployment/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    descriptions = np.array([str(x['description']) for x in request.json])
    descriptions=descriptions.reshape(-1,1)
    prediction = model.predict(descriptions)
    #print(prediction[0])
    #output = prediction[0]

    #return array of objects
    return jsonify(output)

    #data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])

    #return render_template('index.html', prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)
