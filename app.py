#importing libraries
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
from flask import Flask ,render_template ,request

#load model
pac = joblib.load('model/pac_model.sav')
tfidf = joblib.load('model/tfidf_model.sav')

#set up tfidfvectorizor
tfidf_vectorizor=TfidfVectorizer(stop_words='english', max_df=0.7)

app= Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        x=request.form['news']
        tfidf_test=tfidf.transform([x,])
        y_pred=pac.predict(tfidf_test)
        result=str(y_pred).strip("['']")
        return render_template('index.html',result=result)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)