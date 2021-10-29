from telegram.ext import Updater, CommandHandler
import os
from posts import Posts


PORT  = int(os.environ.get('PORT', 8443))
telegram_bot_token = os.environ.get('telegram_bot_token')
        
def main():
    p = Posts()
    updater = Updater(token=telegram_bot_token, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler('recent', p.recent_posts))
    dispatcher.add_handler(CommandHandler('start', p.start))
    dispatcher.add_handler(CommandHandler('feedback', p.feedback))
    dispatcher.add_handler(CommandHandler('links', p.get_links))
    dispatcher.add_handler(CommandHandler('stat', p.get_stat))
    dispatcher.add_handler(CommandHandler('names', p.proper_names))
    '''
    updater.start_webhook(listen="0.0.0.0",
            port=PORT, url_path = telegram_bot_token,
            webhook_url = "https://sentiment-bot-pulse.herokuapp.com/" + telegram_bot_token)
    '''
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
