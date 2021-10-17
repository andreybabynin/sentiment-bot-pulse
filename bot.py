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
plt.style.use("dark_background")
nltk.download('stopwords')
nltk.download('punkt')

with open('./auth.json', 'r') as f:
    auth_dic = json.load(f)
    f.close()

class Parser(object):
    def __init__(self):
        #self.df = pd.DataFrame()
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
        #self.current_ticker = None
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
        plt.savefig('./{}.png'.format(user))


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
        /recent [TICKER] - extract recent posts and sentiment (max 500 symbols) \n
        /stat [TICKER] - statistics about comments \n
        /links [TICKER] - graph of linked stocks based on frequency in recent coments \n
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
    def recent_posts(self, update, context):
        user = update.message.from_user.username
        for i in range(3):
            text = self.user_dic[user]['df'].at[i, 'text'][:500]
            sent, emodji = self.user_dic[user]['df'].at[i, 'sentiment'], self.emodji[self.user_dic[user]['df'].at[i, 'sentiment']]
            text = '{} {}:  \n {}'.format(emodji, sent, text) 
            context.bot.send_message(chat_id=update.effective_chat.id, text=text) 
     
    def get_links(self, update, context):
        user = update.message.from_user.username
        update.message.reply_text('Stat for {}'.format(self.user_dic[user]['current_ticker']))

    @ticker_decorator
    def get_stat(self, update, context):
        user = update.message.from_user.username
        if type(self.user_dic[user]['df']) != type(None):
            self.pie_chart(self.user_dic[user]['df'], user)
            context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('./{}.png'.format(user), "rb"))
        

    def feedback(self, update, context):
        user, text = update.message.from_user.username, update.message.text[10:]
        context.bot.send_message(chat_id=auth_dic['MY_ID'], text = """from user: {}, message: {} \n #FEED""".format(user, text))
        update.message.reply_text('Thank you, {}, for your feedback!'.format(user))
    
        
def main():
    PORT  = int(os.environ.get('PORT', 8443))
    
    updater = Updater(token=auth_dic['telegram_bot_token'], use_context=True)
    dispatcher = updater.dispatcher
    p = Posts()
    dispatcher.add_handler(CommandHandler('recent', p.recent_posts))
    dispatcher.add_handler(CommandHandler('start', p.start))
    dispatcher.add_handler(CommandHandler('feedback', p.feedback))
    dispatcher.add_handler(CommandHandler('links', p.get_links))
    dispatcher.add_handler(CommandHandler('stat', p.get_stat))
    
    updater.start_webhook(listen="0.0.0.0",
            port=PORT, url_path = auth_dic['telegram_bot_token'],
            webhook_url = "https://sentiment-bot-pulse.herokuapp.com/" + auth_dic['telegram_bot_token'])
    
    #updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
    