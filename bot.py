import requests as re
from telegram.ext import Updater, CommandHandler
import json
import os

PORT = int(os.environ.get('PORT', '8443'))
telegram_bot_token = '1284331928:AAFbIt-HNuEhX1uDExXgVyXOvY_PgYXBkxo'

def recent_posts(update, context):

    if len(context.args) != 1:
        context.bot.send_message(chat_id=update.effective_chat.id, text='Ticker is not provided or too much')
    else:
        ticker = context.args[0]
        link='https://www.tinkoff.ru/api/invest-gw/social/v1/post/instrument/{}?limit=30&appName=invest&platform=web'.format(ticker)
        response = re.get(link)
        
        if response.status_code == 200:
            dic = json.loads(response.text)
            items = dic['payload']['items']
            for i in range(3):
                text = items[i]['text']
                min_v = min(len(text), 100)
                context.bot.send_message(chat_id=update.effective_chat.id, text=text[:min_v]) 
        else: 
            context.bot.send_message(chat_id=update.effective_chat.id, text='Try later')


if __name__ == '__main__':
    updater = Updater(token=telegram_bot_token, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler('recent', recent_posts))
    
    updater.start_webhook(listen="0.0.0.0",
                      port=PORT,
                      url_path=telegram_bot_token,
                      webhook_url="https://sentiment-bot-pulse.herokuapp.com/" + telegram_bot_token)
    
    updater.idle()