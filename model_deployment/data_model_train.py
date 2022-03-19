#import all packagaes
import pandas as pd
from sklearn.feature_extraction import _stop_words
import nltk
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score,hamming_loss
from skmultilearn.problem_transform import BinaryRelevance

#cleaning text methods
stopwords = nltk.corpus.stopwords.words('english')
def clean_text(text, checkSkill):
    text = ' '.join([word for word in text.split() if word not in stopwords])
    if(checkSkill):
      #keep symbols as it is part of skills
      text = ''.join([word for word in text if word not in string.punctuation.replace("&", "").replace("/", "")])
    else:
      text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    return text

#read in skills and related areas
with open("skills.txt", "r") as skills_file:
  og_categories = skills_file.read().split('\n')
  categories = pd.DataFrame(og_categories)
  categories = categories[0].apply(lambda x: clean_text(str(x.strip()), True))

#clean text
with open('sample_data.csv', encoding='utf-8') as sample_data:
  file = pd.read_csv(sample_data, delimiter=None, header=None)
  
  descriptions = file[0].apply(lambda x: clean_text(str(x.strip()), False))
  skills = file[1].apply(lambda x: clean_text(str(x.strip()), True))

  clean_file = pd.DataFrame()
  clean_file.insert(0, 'description', descriptions)
 
  #create one-hot encoding file in clean_file
  for i, each in enumerate(categories):
    temp_array=[]
    for j in range(len(skills)):
      if(each in skills[j]):
        temp_array.append(1)
      else:
        temp_array.append(0)
    df = pd.DataFrame(temp_array, columns=[each])
    clean_file=pd.concat([clean_file, df], axis=1)

# Instantiate the CountVectorizer method
tfidf = TfidfVectorizer()
Xfeatures = tfidf.fit_transform(clean_file['description']).toarray()

# split into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(Xfeatures,clean_file[categories],test_size=0.2, random_state=959)

model = BinaryRelevance(MultinomialNB())
model.fit(X_train,y_train)
BinaryRelevance(classifier=MultinomialNB(alpha=1.0, class_prior=None,
                                         fit_prior=True),
                require_dense=[True, True])
prediction = model.predict(X_test)
prediction.toarray()

pickle.dump(model, open('model_deployment/model.pkl', "wb"))