# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from sentiment_classifier import SentimentClassifier
import time
from flask import Flask, render_template, request
from datetime import datetime
from sklearn.externals import joblib
app = Flask(__name__)

print('Preparing classifier')
start_time = time.time()
classifier = SentimentClassifier()
print('Classifier ready')
print(time.time() - start_time, 'seconds')

def timestamp():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

@app.route('/', methods=['POST', 'GET'])
def index_page(text="", prediction_message="", color_class=""):

    if request.method == 'POST':
        text = request.form['text']
        prediction_message, color_class = classifier.get_predictions([text])[0]

    return render_template('index.html', text=text, prediction_message=prediction_message, color_class=color_class)

@app.route('/testset', methods=['GET'])
def test_set_page():
    test_data = joblib.load('test_data.pkl')
    test_data = tuple(('+' if label else '-', review, 'positive' if label else 'negative') for label, review in test_data)
    return render_template('testset.html', reviews=test_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False)
