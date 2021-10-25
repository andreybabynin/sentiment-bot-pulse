import requests as re
from telegram.ext import Updater, CommandHandler
import json
import joblib
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string
import re as r
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import networkx as nx
from itertools import combinations
from wordcloud import WordCloud
import io
from natasha import (
    MorphVocab,
    Doc,
    NewsNERTagger,
    NewsEmbedding,
    Segmenter
)

plt.style.use("dark_background")
nltk.download('stopwords')
nltk.download('punkt')
PORT  = int(os.environ.get('PORT', 8443))
telegram_bot_token = os.environ.get('telegram_bot_token')
MY_ID = int(os.environ.get('MY_ID'))

class Parser(object):
    def __init__(self):
        self.base_link = 'https://www.tinkoff.ru/api/invest-gw/social/v1/post/instrument/{}?limit=30&appName=invest&platform=web'
        
    def stock_parser(self, ticker, length = 1):
        cursor = 0
        #columns
        text  = []
        time_stamp = []
        likes = []
        author = []
        comments_count = []
        ids = []
        author_ids = []
        mentioned_count = []
        mentioned = []
        
        def format_time(time_stamp):
            return datetime.strptime(time_stamp[:19], '%Y-%m-%dT%H:%M:%S')
        
        for i in range(length):
            if i==0:
                link=self.base_link.format(ticker)
            else:
                link=(self.base_link +'&cursor={}').format(ticker, cursor)
            response = re.get(link)
                        
            if response.status_code == 200:
                dic = json.loads(response.text)
                if dic['status'] == 'Ok':
                    # next cursor
                    cursor = dic['payload']['nextCursor']
                    items = dic['payload']['items']
                    for item in items:
                        author.append(item['owner']['nickname'])
                        author_ids.append(item['owner']['id'])
                        text.append(item['text'])
                        likes.append(item['likesCount'])
                        time_stamp.append(format_time(item['inserted']))
                        comments_count.append(item['commentsCount'])
                        ids.append(item['id'])
                        mentioned_count.append(len(item['content']['instruments']))
                        mentioned.append(' '.join([t['ticker'] for t in item['content']['instruments']])) 
                        
                    return 1, pd.DataFrame({'comment_id':ids,
                                               'ticker': [ticker]*len(author),
                                               'datetime_comment':time_stamp,
                                               'datetime_grab': [datetime.now()+timedelta(hours=3)]*len(author),
                                               'likes_count': likes, 
                                               'comments_count': comments_count,
                                               'author_id': author_ids, 
                                               'author_name': author, 
                                               'text': text,
                                               'mentioned_count': mentioned_count,
                                                'mentioned': mentioned})
                    
                else:
                    return -1, None
            else:
                return -2, None
                

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
    
    def _score_posts(self, df):
        df['sentiment'] = df['text'].apply(lambda x: self._test_model(x, proba=False)[0]).map(self.MAP_V)

class Graphics():
    def __init__(self):
        self.dic_color =  {'Bullish': 'green', 'Neutral': 'grey', 'Bearish': 'red'}
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.emb)
        self.morph_vocab = MorphVocab()
    
    def pie_chart(self, df, user):
        s = df['sentiment'].value_counts()
        fig, ax = plt.subplots()
        labels = s.index.tolist()
        wp = { 'linewidth' : 1, 'edgecolor' : "black" }
        wedges, texts, autotexts = ax.pie(s.values, autopct = '%.0f%%', explode = [0.03]*3, 
                                  labels = labels, shadow = True,
                                  colors = [self.dic_color[i] for i in labels], wedgeprops = wp)
        ax.legend(wedges, labels,
              title ="Sentiment",
              loc ='upper right',
              bbox_to_anchor =(0.8, 0.1, 0.5, 1))
        plt.setp(autotexts, size = 10, weight ="bold")
        plt.title('{} sentiment for the last 30 posts'.format(df.at[0, 'ticker']))
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        return buf

    def link_chart(self, df, user):
        network_df = self._find_connections(df)
        fig = plt.figure()
        ax = plt.gca()
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        network1 = nx.from_pandas_edgelist(network_df, 
                                           'Ticker', 'Target', edge_attr='Weight', create_using=nx.Graph)
        pos = nx.spring_layout(network1, k=0.55)
        options = {
            "node_size": 1,
            "width" : 0.5,
            "style": 'dashed',
            'edge_color': 'blue',
            "edge_vmin": 0,
            "edge_vmax": 5,
            "font_size": 10, 
            "with_labels": True}
        
        plt.title('Connections between stocks in posts about {}'.format(df.at[1, 'ticker']))
        nx.drawing.nx_pylab.draw_networkx(network1, pos = pos, **options)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        return buf
        
    def _find_connections(self, df):
        dic_m = {}
        for row in df['mentioned'].tolist():
            list1 = row.split(' ')
            list1.remove(df.at[0, 'ticker'])
            if list1 != None:
                comb = list(combinations(list1, 2))
                for i in comb:
                    if (i in dic_m.keys()) or ((i[1], i[0]) in dic_m.keys()):
                        try:
                            dic_m[i] += 1
                        except: dic_m[(i[1], i[0])] += 1
                    else:
                        dic_m[i] = 1
        list1 = []
        for k,v in dic_m.items():
            list1.append([k[0], k[1], v])
        return pd.DataFrame(list1, columns = ['Ticker', 'Target', 'Weight'])
    
    def _proper_names(self, df):
        entr = set()
        for row in df.text.values:
            clean_text = self._clean_text(row)
            doc = Doc(clean_text)
            doc.segment(self.segmenter)
            doc.tag_ner(self.ner_tagger)
            for span in doc.spans:
                span.normalize(self.morph_vocab)
                if span.type== 'ORG':
                    entr.add(r.sub('[0-9.,!?]*$%^', '', span.text).rstrip())
        return entr
    
    def proper_word_chart(self, df, user):
        entr = list(self._proper_names(df))
        string = entr[0]
        for e in entr[1:]:
            string = string+',' + e
        wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue', width=400, height=400)
        wordcloud.generate(string)
        img = wordcloud.to_image()
        
        buf = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        return buf
        
class Posts(Sentiment, Parser, Graphics):
    def __init__(self):
        Sentiment.__init__(self)
        Parser.__init__(self)
        Graphics.__init__(self)
        self.user_dic = {} #save responses from users
    
    #isolate data between users
    @staticmethod
    def add_user_to_dic(user, user_dic):
        if user not in user_dic.keys():
            user_dic[user] = {'current_ticker': None, 'df': None}
    
    @classmethod
    def start(cls, update, context):
        text_start = """This BOT can do several things: \n
        /recent [TICKER] - extract recent posts and sentiment (max 500 symbols)
        /stat [TICKER] - statistics about comments
        /links [TICKER] - graph of linked stocks based on frequency in recent comments
        /names [TICKER] - visualize proper names in posts
        /feedback - write about your experience"""
    
        context.bot.send_message(chat_id = update.effective_chat.id, text = text_start) 
    
    def grab_data(self, update, context):
        user = update.message.from_user.username
        res, self.user_dic[user]['df'] = self.stock_parser(self.user_dic[user]['current_ticker'])
        if res > 0:
            self._score_posts(self.user_dic[user]['df'])
        else: 
            context.bot.send_message(chat_id=update.effective_chat.id, text='Try later, Service is not responding')
            
    def ticker_decorator(func):
        
        def wrapper(self, update, context):
            user = update.message.from_user.username
            Posts.add_user_to_dic(user, self.user_dic)
        
            if len(context.args) > 1:
                context.bot.send_message(chat_id=update.effective_chat.id, text='Too many tickers')
            elif len(context.args) == 0 and self.user_dic[user]['current_ticker'] != None:
                context.bot.send_message(chat_id=update.effective_chat.id, 
                                         text='Data for the last ticker provided: {}'.format(self.user_dic[user]['current_ticker']))
                func(self, update, context)
            elif len(context.args) == 0:
                context.bot.send_message(chat_id=update.effective_chat.id, text='Provide ticker')
            else:
                ticker = context.args[0].upper()
                if self.user_dic[user]['current_ticker'] != ticker:
                    self.user_dic[user]['current_ticker'] = ticker
                    self.grab_data(update, context)
                    func(self, update, context)
                else:
                    func(self, update, context)
        return wrapper
    
    @ticker_decorator
    def proper_names(self, update, context):
        user = update.message.from_user.username
        df = self.user_dic[user]['df']
        
        buf = self.proper_word_chart(df, user)
        buf.seek(0)
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=buf)
        buf.close()
    
    @ticker_decorator
    def recent_posts(self, update, context):
        user = update.message.from_user.username
        for i in range(3):
            text = self.user_dic[user]['df'].at[i, 'text'][:500]
            sent, emodji = self.user_dic[user]['df'].at[i, 'sentiment'], self.emodji[self.user_dic[user]['df'].at[i, 'sentiment']]
            text = '{} {}:  \n\n {}'.format(emodji, sent, text) 
            context.bot.send_message(chat_id=update.effective_chat.id, text=text) 
     
    @ticker_decorator    
    def get_links(self, update, context):
        user = update.message.from_user.username
        buf = self.link_chart(self.user_dic[user]['df'], user)
        buf.seek(0)
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=buf)
        buf.close()

    @ticker_decorator
    def get_stat(self, update, context):
        user = update.message.from_user.username
        buf = self.pie_chart(self.user_dic[user]['df'], user)
        buf.seek(0)
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=buf)
        buf.close()

    def feedback(self, update, context):
        user, text = update.message.from_user.username, update.message.text[10:]
        
        context.bot.send_message(chat_id=MY_ID, text = """from user: {}, message: {} \n\n #FEED""".format(user, text))
        update.message.reply_text('Thank you, {}, for your feedback!'.format(user))
    
        
def main():
    updater = Updater(token=telegram_bot_token, use_context=True)
    dispatcher = updater.dispatcher
    p = Posts()
    dispatcher.add_handler(CommandHandler('recent', p.recent_posts))
    dispatcher.add_handler(CommandHandler('start', p.start))
    dispatcher.add_handler(CommandHandler('feedback', p.feedback))
    dispatcher.add_handler(CommandHandler('links', p.get_links))
    dispatcher.add_handler(CommandHandler('stat', p.get_stat))
    dispatcher.add_handler(CommandHandler('names', p.proper_names))
    
    updater.start_webhook(listen="0.0.0.0",
            port=PORT, url_path = telegram_bot_token,
            webhook_url = "https://sentiment-bot-pulse.herokuapp.com/" + telegram_bot_token)
    
    #updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
