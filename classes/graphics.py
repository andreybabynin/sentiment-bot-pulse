from natasha import (
    MorphVocab,
    Doc,
    NewsNERTagger,
    NewsEmbedding,
    Segmenter
)
import matplotlib.pyplot as plt
import io
import networkx as nx
from itertools import combinations
import pandas as pd
import re as r
from wordcloud import WordCloud

plt.style.use("dark_background")


class GraphicsTools():
    def __init__(self):
        self.dic_color =  {'Bullish': 'green', 'Neutral': 'grey', 'Bearish': 'red'}
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.emb)
        self.morph_vocab = MorphVocab()
    
    def pie_chart(self, context):
        df = context.user_data['df']
        
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

    def link_chart(self, context):
        
        network_df = self._find_connections(context)
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
        
        plt.title('Connections between stocks in posts about {}'.format(context.user_data['current_ticker']))
        nx.drawing.nx_pylab.draw_networkx(network1, pos = pos, **options)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        return buf
    
    @staticmethod
    def create_df_from_edge_dic(dic_graph_edges):
        temp_list = []
        for k, v in dic_graph_edges.items():
            temp_list.append([k[0], k[1], v])
        return pd.DataFrame(temp_list, columns = ['Ticker', 'Target', 'Weight'])
        
    def _find_connections(self, context):
        df = context.user_data['df']
        dic_graph_edges = {}
        for row in df['mentioned'].tolist():
            list1 = row.split(' ')
            list1.remove(df.at[0, 'ticker'])
            if list1 != None:
                comb = list(combinations(list1, 2))
                for i in comb:
                    if (i in dic_graph_edges.keys()) or ((i[1], i[0]) in dic_graph_edges.keys()):
                        try:
                            dic_graph_edges[i] += 1
                        except: dic_graph_edges[(i[1], i[0])] += 1
                    else:
                        dic_graph_edges[i] = 1
        return create_df_from_edge_dic(dic_graph_edges)
    
    def _proper_names(self, context):
        df = context.user_data['df']
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
    
    def proper_word_chart(self, context):
        
        entr = list(self._proper_names(context))
        string = entr[0]
        for e in entr[1:]:
            string = string+',' + e
        wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue', width=400, height=400)
        wordcloud.generate(string)
        img = wordcloud.to_image()
        buf = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        return buf