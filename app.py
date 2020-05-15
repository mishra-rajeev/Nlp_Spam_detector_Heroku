# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:28:14 2020

@author: Rajeev Mishra
"""

from flask import Flask, render_template,url_for,request
import random
import pickle
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

filename = 'spam_classifier.pickle'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('predict_classifier.pickle','rb'))
app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
			message = request.form['message']
			data = ['message']
			m =' '.join(map(str, data))			
			my_prediction =cv
	return render_template('result.html',prediction = my_prediction)


if __name__ ==  '__main__':
	app.run(debug=True)