import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
#load model
model = pickle.load(open('model_deployment/model.pkl', "wb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    descriptions = [int(x) for x in request.form.values()]
    final_descriptions = [np.array(descriptions)]
    prediction = model.predict(final_descriptions)

    return render_template('index.html', prediction = prediction)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
