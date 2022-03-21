from ast import Str
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import nltk, string, pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
#load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

stopwords = nltk.corpus.stopwords.words('english')
def clean_text(text):
    text = ' '.join([word for word in text.split() if word not in stopwords])
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    return text

with open("skills.txt", "r") as skills_file:
  og_categories = skills_file.read().split('\n')
  categories = pd.DataFrame(og_categories)
  categories = categories[0].apply(lambda x: clean_text(str(x.strip())))

@app.route('/')
def home():
    return None

@app.route('/predict', methods=['POST'])
def predict():
    final_array=[]
    descriptions = pd.DataFrame([str(x['description']) for x in request.json])
    
    #clean text
    descriptions = descriptions[0].apply(lambda a: clean_text(str(a.strip())))
    vectorizer_transform = vectorizer.transform(descriptions).toarray()
    prediction = model.predict(vectorizer_transform)
    
    #convert binary numbers to category values
    for j, each in enumerate(prediction.toarray()):
        temp_array=[]
        for i in range(len(each)):
            if each[i] == 1:
                temp_array.append(categories[i])
        final_array.append({"id": request.json[j]['id'], "skills": temp_array})
    
    #return array of objects
    return jsonify(final_array)

if __name__ == "__main__":
    app.run(debug=True)
