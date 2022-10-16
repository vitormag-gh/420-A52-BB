# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:25:53 2022

@author: riskf
"""
from flask import Flask, request
from flask import jsonify, render_template
import pickle
import numpy as np
import os
#from sklearn.feature_extraction.text import CountVectorizer 

pathFile='C:\\Users\\riskf\\OneDrive\\Documents\\Courses\\AEC Spécialiste en intelligence artificielle\\Courses\\4 - 420-A52-BB - Algorithmes d’apprentissage supervisé\\Projet_session\\'
os.chdir(pathFile)

app = Flask(__name__, template_folder=pathFile)

model_pickle = pickle.load(open('voting.pkl', 'rb'))
selector=pickle.load(open('vector.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    final_features = request.form['message']
    final_features = selector.transform([final_features]).toarray()
    prediction = model_pickle.predict(final_features)
    output = prediction[0]
    

    return render_template('home.html', prediction_text='The text shows signs of: {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model_pickle.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)


#if __name__ == "__main__":
 #   app.run(host="127.0.0.1", port=8080, debug=True)