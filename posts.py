import os
from classes.parser import Parser
from classes.sentiment import Sentiment
from classes.graphics import GraphicsTools

MY_ID = int(os.environ.get('MY_ID'))

class Posts(Sentiment, Parser, GraphicsTools):
    def __init__(self):
        Sentiment.__init__(self)
        Parser.__init__(self)
        GraphicsTools.__init__(self)
    
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
        res = self.stock_parser(context)

        if res > 0:

            self._score_posts(context)
        else: 
            context.bot.send_message(chat_id=update.effective_chat.id, text='Try later, Service is not responding')
            
    def ticker_decorator(func):
        
        def wrapper(self, update, context):

            if 'current_ticker' not in context.user_data.keys():
                context.user_data['current_ticker'] = None            
            
            if len(context.args) > 1:
                context.bot.send_message(chat_id=update.effective_chat.id, text='Too many tickers')

            elif len(context.args) == 0 and context.user_data['current_ticker'] != None:

                context.bot.send_message(chat_id=update.effective_chat.id, 
                                         text='Data for the last ticker provided: {}'.format(context.user_data['current_ticker']))
                func(self, update, context)
            elif len(context.args) == 0:
                context.bot.send_message(chat_id=update.effective_chat.id, text='Provide ticker')
            else:
                ticker = context.args[0].upper()
                
                if ticker !=context.user_data['current_ticker']:
                        
                    context.user_data['current_ticker'] = ticker
                    self.grab_data(update, context)
                    func(self, update, context)
                else:
                    func(self, update, context)
        return wrapper
    
    @ticker_decorator
    def proper_names(self, update, context):
        
        buf = self.proper_word_chart(context)
        buf.seek(0)
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=buf)
        buf.close()
    
    @ticker_decorator
    def recent_posts(self, update, context):

        for i in range(3):
            text = context.user_data['df'].at[i, 'text'][:500]
            sent, emodji = context.user_data['df'].at[i, 'sentiment'], self.emodji[context.user_data['df'].at[i, 'sentiment']]
            text = '{} {}:  \n\n {}'.format(emodji, sent, text) 
            context.bot.send_message(chat_id=update.effective_chat.id, text=text) 
     
    @ticker_decorator    
    def get_links(self, update, context):
        buf = self.link_chart(context)
        buf.seek(0)
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=buf)
        buf.close()

    @ticker_decorator
    def get_stat(self, update, context):
        buf = self.pie_chart(context)
        buf.seek(0)
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=buf)
        buf.close()

    def feedback(self, update, context):
        user, text = update.message.from_user.username, update.message.text[10:]
        
        context.bot.send_message(chat_id=MY_ID, text = """from user: {}, message: {} \n\n #FEED""".format(user, text))
        update.message.reply_text('Thank you, {}, for your feedback!'.format(user))