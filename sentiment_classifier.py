# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from sklearn.externals import joblib
from nltk import word_tokenize

import nltk
nltk.download('punkt')


class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load('model.pkl')
        self.cls_dict = {0: ('отрицательный', 'negative'), 1: ('положительный', 'positive'), -1: ('ошибка', 'error')}

    @staticmethod
    def get_certainty(probability):
        if probability < 0.55:
            return 'возможно'
        if probability < 0.7:
            return 'вероятно'
        if probability > 0.95:
            return 'определённо'
        else:
            return ''

    def predict_list(self, texts):
        try:
            return self.model.predict(texts), self.model.predict_proba(texts)
        except:
            print('prediction error')
            return None

    def clean_sentence(self, sentence, keep_digits=True):
        return ' '.join([word.lower() for word in word_tokenize(sentence) if word.isalpha() or keep_digits])

    def get_predictions(self, texts):
        texts = [self.clean_sentence(sentence) for sentence in texts]
        preds = self.predict_list(texts)
        print(preds)
        return [('{0} {1}'.format(self.get_certainty(pred[1].max()), self.cls_dict[pred[0]][0]), self.cls_dict[pred[0]][1]) for pred in zip(*preds)]
