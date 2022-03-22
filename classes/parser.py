import requests as re
from datetime import datetime, timedelta
import json
import pandas as pd

CORRECT_RESPONSE = 1
SERVER_NOT_RESPONDING = -2
DATA_IS_CORRUPTED = -1


class Parser(object):
    def __init__(self):
        self.base_link = 'https://www.tinkoff.ru/api/invest-gw/social/v1/post/instrument/{}?limit=30&appName=invest&platform=web'
        self.time_part = 19
           
    def stock_parser(self, context, length = 1):
        ticker = context.user_data['current_ticker']
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
            return datetime.strptime(time_stamp[:self.time_part], '%Y-%m-%dT%H:%M:%S')
        
        def get_data(dic, context):
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
                    
            context.user_data['df'] = pd.DataFrame({'comment_id':ids,
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
            return cursor
        
        for i in range(length):
            if i==0:
                link=self.base_link.format(ticker)
            else:
                link=(self.base_link +'&cursor={}').format(ticker, cursor)
            response = re.get(link)
                        
            if response.status_code == 200:
                dic = json.loads(response.text)
                
                if dic['status'] == 'Ok':
                    cursor = get_data(dic, context)
                    return CORRECT_RESPONSE
                    
                else:
                    return DATA_IS_CORRUPTED
            else:
                return SERVER_NOT_RESPONDING