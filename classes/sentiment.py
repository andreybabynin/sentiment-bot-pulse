import re as r
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import joblib
import string

nltk.download('stopwords')
nltk.download('punkt')


class Sentiment(object):
    def __init__(self):        
        self.russsian_stop = stopwords.words('russian')
        self.snowball = SnowballStemmer(language='russian')
        self.vectorizer = joblib.load('./models/vectorizer.joblib')
        self.model = joblib.load('./models/model_3_classes.joblib')
        self.MAP_V = {-1: 'Bearish', 0: 'Neutral', 1: 'Bullish'}
        self.emodji = {'Bullish': "\N{slightly smiling face}", 'Bearish': "\N{slightly smiling face}", 'Neutral': "\N{neutral face}"}
        
    def _remove_hastags(self, text):
        hastags = list(set(r.findall('\#\S*', text)))
        if len(hastags)>0:
            for hash in hastags:
                text = text.replace(hash, ' ')
        return text

    def _remove_tickers(self, text):
        tickers = list(set(r.findall('\$\S*' , text)))
        if len(tickers)>0:
            for ticker in tickers:
                text = text.replace(ticker, ' ')
        return text

    def _remove_symbols(self, text):
        special_sym = "«»*{}$#[]():;?!'+-=_"
        for ch in special_sym:
            text = text.replace(ch, "")
        return text

    def _clean_text(self, text):
        text = self._remove_hastags(text)
        text = self._remove_tickers(text)
        text = self._remove_symbols(text)
        return text

    def _tokens_from_text(self, text):
        cleaned_text = self._clean_text(text)
        
        tokens = word_tokenize(cleaned_text, language='russian')
        tokens_wo_punkt = [i for i in tokens if i not in string.punctuation]
        tokens_wo_stop = [i for i in tokens_wo_punkt if i not in self.russsian_stop]
        tokens_wo_num = [r.sub('[0-9]', '', i) for i in tokens_wo_stop]
        stemmed_tokens = [self.snowball.stem(i) for i in tokens_wo_num]
        stemmed_tokens_2 = ' '.join([i for i in stemmed_tokens if len(i)>2])
    
        return stemmed_tokens_2

    def _test_model(self, text, proba=True):
        tokens = self._tokens_from_text(text)
        x = self.vectorizer.transform([tokens])
        if proba:
            return self.model.predict_proba(x)
        else:
            return self.model.predict(x)
    
    def _score_posts(self, context):
        context.user_data['df']['sentiment'] = context.user_data['df']['text'].apply(lambda x: self._test_model(x, proba=False)[0]).map(self.MAP_V)