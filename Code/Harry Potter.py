#EX04 Applying NLP
#by Akash Bhoite

#Importing the necessary libraries

import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import string
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import pyLDAvis.sklearn
import concurrent.futures
from datetime import datetime

#Importing the books

H1 = []
H2 = []
H3 = []
H4 = []
H5 = []
H6 = []
H7 = []

f1 = open('/Users/akashbhoite/Datasets/(Book I) Harry Potter and the Sorcerer_s Stone.txt')
f2 = open('/Users/akashbhoite/Datasets/(Book II) Harry Potter and the Chamber of Secrets.txt')
f3 =  open('/Users/akashbhoite/Datasets/(Book III) Harry Potter and the Prisoner of Azkaban.txt')
f4 =  open('/Users/akashbhoite/Datasets/(Book IV) Harry Potter and the Goblet of Fire.txt')
f5 =  open('/Users/akashbhoite/Datasets/(Book V) Harry Potter and the Order of the Phoenix.txt')
f6 =  open('/Users/akashbhoite/Datasets/(Book VI) Harry Potter and the Half-Blood Prince.txt')
f7 =  open('/Users/akashbhoite/Datasets/(Book VII) Harry Potter and the Deathly Hallows.txt')
            
for line in f1:
    inner = line.strip().split()
    H1.append(inner)
        
for line in f2:
    inner = line.strip().split()
    H2.append(inner)
        
for line in f3:
    inner = line.strip().split()
    H3.append(inner)
    
for line in f4:
    inner = line.strip().split()
    H4.append(inner)
        
for line in f5:
    inner = line.strip().split()
    H5.append(inner)
        
for line in f6:
    inner = line.strip().split()
    H6.append(inner)

for line in f7:
    inner = line.strip().split()
    H7.append(inner)

#Creating a list of each book

H1 = [item for items in H1 for item in items]
H2 = [item for items in H2 for item in items]
H3 = [item for items in H3 for item in items]
H4 = [item for items in H4 for item in items]
H5 = [item for items in H5 for item in items]
H6 = [item for items in H6 for item in items]
H7 = [item for items in H7 for item in items]


#Adding more context-based stopwords to our stopwords list


more_stp_words = ['harry','potter','mr','mrs','an','am','said','dont','like','dint','know','ive','got'
                 ,'could','see','havent','im','going','looked','through','tell','yeh',
                 'go','back','get','yer','would','never','seen','something','else','next','day','years'
                 ,'didnt','look','thousand','one','ever','even','though','although','every','time','make','sure'
                 ,'told','minutes','minute','hour','where','else','no','oh','ten','year','front','door','wandering','nine',
                 'ten','points','fifty','hundred','twenty','gotten','fell','asleep','yes','trying','find','one','two','three'
                 ,'four','thirteen','clock','give','us','youd','expect','picked','lived','number','twice','youknowwho','think',
                  'hed','harrys','well','around']
stop_words.extend(more_stp_words)


# Function for cleaning the text


def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = " ".join(text)
    text = text.lower()
    text = text.strip()
    text = re.sub('\[.*?\]’', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.replace("’","")
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
        
    return text


# Function for tokenizing and removing stop words


def final_clean(texts):
    toks = word_tokenize(texts)
    stp = [word for word in toks if word not in stop_words]
    #stpr = ' '.join(stp)
    return stp


# Function to find bigrams 


def ngram(text,grams):    
    n_grams_list = []    
    count = 0    
    for token in text[:len(text)-grams+1]:       
        #n_grams_list.append(text[count:count+grams])      
        n_grams_list.append(text[count]+' '+text[count+grams-1])        
        count=count+1      
    return n_grams_list


# Function to find the most common/most occuring words


def most_common(lst, num):
    
    data = Counter(lst)    
    common = data.most_common(num)    
    top_comm = []    
    for i in range (0, num):        
        top_comm.append (common [i][0])    
    return top_comm


# Function for replacing the singular words with bigrams


def chunk_replacement(chunk_list, text):
    """    Connects words chunks in a text by joining them with an underscore.  
    :param chunk_list: word chunks    
    :type chunk_list: list of strings/ngrams   
    :param text: text    
    :type text: string    :return: text with underscored chunks    :type: string    """ 
    text = ' '.join(text)
    for chunk in chunk_list:
            text = text.replace(chunk, chunk.replace(' ', '_'))    
    return text


# Function for creating wordclouds


def show_wordcloud(data):
    data = ' '.join(data)
    wc = WordCloud(background_color="white", colormap="viridis",
               max_font_size=150, random_state=42,max_words=4000)
    wc.generate(data)
    plt.axis('off')
    plt.imshow(wc)
    plt.show()


# Function for Sentiment Analysis


def sentiment_analyzer(data,bookname):
    analyzer = SentimentIntensityAnalyzer()
    # vader needs strings as input. Transforming the list into string
    data_str = ' '.join(data)
    vad_sentiment = analyzer.polarity_scores(data_str)
    pos = vad_sentiment ['pos']
    neg = vad_sentiment ['neg']
    neu = vad_sentiment ['neu']
    print ('\nThe following is the distribution of the sentiment for the file -',bookname)
    print ('\n--- It is positive for', '{:.1%}'.format(pos))
    print ('\n--- It is negative for', '{:.1%}'.format(neg))
    print ('\n--- It is neutral for', '{:.1%}'.format(neu), '\n')


# Functions for printing keywords for each topic

def selected_topics(model, vectorizer, top_n=10):    
    for idx, topic in enumerate(model.components_):        
        print("Topic %d:" % (idx))        
        print([(vectorizer.get_feature_names()[i], topic[i])                        
               for i in topic.argsort()[:-top_n - 1:-1]])
        print('\n')


#Creating a smaller name for all the books

book1 = '(Book I) Harry Potter and the Sorcerer_s Stone'
book2 = '(Book II) Harry Potter and the Chamber of Secrets'
book3 = '(Book III) Harry Potter and the Prisoner of Azkaban'
book4 = '(Book IV) Harry Potter and the Goblet of Fire'
book5 = '(Book V) Harry Potter and the Order of the Phoenix'
book6 = '(Book VI) Harry Potter and the Half-Blood Prince'
book7 = '(Book VII) Harry Potter and the Deathly Hallows'


#Defining parameters for Topic detection


text_file = book1
stp_file = stop_words
word_min_len = 2
num_topics = 5


# Cleaning the books


H1_clean1 = clean_text(H1)
H2_clean1 = clean_text(H2)
H3_clean1 = clean_text(H3)
H4_clean1 = clean_text(H4)
H5_clean1 = clean_text(H5)
H6_clean1 = clean_text(H6)
H7_clean1 = clean_text(H7)


#Tokenizing all the books and removing stopwords


H1_clean = final_clean(H1_clean1)
H2_clean = final_clean(H2_clean1)
H3_clean = final_clean(H3_clean1)
H4_clean = final_clean(H4_clean1)
H5_clean = final_clean(H5_clean1)
H6_clean = final_clean(H6_clean1)
H7_clean = final_clean(H7_clean1)


#Creating bigrams of all the books 


H1_bigrams = ngram(H1_clean,2)
H2_bigrams = ngram(H2_clean,2)
H3_bigrams = ngram(H3_clean,2)
H4_bigrams = ngram(H4_clean,2)
H5_bigrams = ngram(H5_clean,2)
H6_bigrams = ngram(H6_clean,2)
H7_bigrams = ngram(H7_clean,2)


#Top 10 bigrams from each of the books

H1_bigrams_10 = most_common(H1_bigrams,10)
H2_bigrams_10 = most_common(H2_bigrams,10)
H3_bigrams_10 = most_common(H3_bigrams,10)
H4_bigrams_10 = most_common(H4_bigrams,10)
H5_bigrams_10 = most_common(H5_bigrams,10)
H6_bigrams_10 = most_common(H6_bigrams,10)
H7_bigrams_10 = most_common(H7_bigrams,10)

#Printing the top 10 Bigrams for each book

print('The top 10 bigrams for Book 1 are: \n',H1_bigrams_10,'\n')
print('The top 10 bigrams for Book 2 are: \n',H2_bigrams_10,'\n')
print('The top 10 bigrams for Book 3 are: \n',H3_bigrams_10,'\n')
print('The top 10 bigrams for Book 4 are: \n',H4_bigrams_10,'\n')
print('The top 10 bigrams for Book 5 are: \n',H5_bigrams_10,'\n')
print('The top 10 bigrams for Book 6 are: \n',H6_bigrams_10,'\n')
print('The top 10 bigrams for Book 7 are: \n',H7_bigrams_10,'\n')

#Chunk replacement
#Replacing single words for the top 10 bigrams 

H1_chunk = chunk_replacement(H1_bigrams_10,H1_clean)
H2_chunk = chunk_replacement(H2_bigrams_10,H2_clean)
H3_chunk = chunk_replacement(H3_bigrams_10,H3_clean)
H4_chunk = chunk_replacement(H4_bigrams_10,H4_clean)
H5_chunk = chunk_replacement(H5_bigrams_10,H5_clean)
H6_chunk = chunk_replacement(H6_bigrams_10,H6_clean)
H7_chunk = chunk_replacement(H7_bigrams_10,H7_clean)


#Sentiment analysis of all books (This takes the most amount of time)

sentiment_analyzer(H1_clean,book1)
print('\n')
sentiment_analyzer(H2_clean,book2)
print('\n')
sentiment_analyzer(H3_clean,book3)
print('\n')
sentiment_analyzer(H4_clean,book4)
print('\n')
sentiment_analyzer(H5_clean,book5)
print('\n')
sentiment_analyzer(H6_clean,book6)
print('\n')
sentiment_analyzer(H7_clean,book7)


#Creating wordclouds for each book

print('Wordcloud for',book1)
show_wordcloud(H1_clean)
print('\n Wordcloud for',book2)
show_wordcloud(H2_clean)
print('\n Wordcloud for',book3)
show_wordcloud(H3_clean)
print('\n Wordcloud for',book4)
show_wordcloud(H4_clean)
print('\n Wordcloud for',book5)
show_wordcloud(H5_clean)
print('\n Wordcloud for',book6)
show_wordcloud(H6_clean)
print('\n Wordcloud for',book7)
show_wordcloud(H7_clean)


#Topic detection for Book 1


vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(H1_clean)

lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online',verbose=False)
data_lda = lda.fit_transform(data_vectorized)

# Keywords for topics clustered by Latent Dirichlet Allocation
print('##############THE 5 TOPICS FOR BOOK 1 ARE AS FOLLOWS###############')
print('\nLDA Model:')
selected_topics(lda, vectorizer)
# visualizing the results. An html interactive file will be created
dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(dash, 'LDA_Visualization1.html')


#Topic detection for Book 2

vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(H2_clean)

lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online',verbose=False)
data_lda = lda.fit_transform(data_vectorized)

# Keywords for topics clustered by Latent Dirichlet Allocation
print('##############THE 5 TOPICS FOR BOOK 2 ARE AS FOLLOWS###############')
print('\nLDA Model:')
selected_topics(lda, vectorizer)
# visualizing the results. An html interactive file will be created
dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(dash, 'LDA_Visualization2.html')


# Topic detection for Book 3


vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(H3_clean)

lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online',verbose=False)
data_lda = lda.fit_transform(data_vectorized)

# Keywords for topics clustered by Latent Dirichlet Allocation
print('##############THE 5 TOPICS FOR BOOK 3 ARE AS FOLLOWS###############')
print('\nLDA Model:')
selected_topics(lda, vectorizer)
# visualizing the results. An html interactive file will be created
dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(dash, 'LDA_Visualization3.html')


# Topic Detection for Book 4


vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(H4_clean)

lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online',verbose=False)
data_lda = lda.fit_transform(data_vectorized)

# Keywords for topics clustered by Latent Dirichlet Allocation
print('##############THE 5 TOPICS FOR BOOK 4 ARE AS FOLLOWS###############')
print('\nLDA Model:')
selected_topics(lda, vectorizer)
# visualizing the results. An html interactive file will be created
dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(dash, 'LDA_Visualization4.html')


# Topic Detection for Book 5


vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(H5_clean)

lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online',verbose=False)
data_lda = lda.fit_transform(data_vectorized)

# Keywords for topics clustered by Latent Dirichlet Allocation
print('##############THE 5 TOPICS FOR BOOK 5 ARE AS FOLLOWS###############')
print('\nLDA Model:')
selected_topics(lda, vectorizer)
# visualizing the results. An html interactive file will be created
dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(dash, 'LDA_Visualization5.html')


# Topic Detection for Book 6


vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(H6_clean)

lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online',verbose=False)
data_lda = lda.fit_transform(data_vectorized)

# Keywords for topics clustered by Latent Dirichlet Allocation
print('##############THE 5 TOPICS FOR BOOK 6 ARE AS FOLLOWS###############')
print('\nLDA Model:')
selected_topics(lda, vectorizer)
# visualizing the results. An html interactive file will be created
dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(dash, 'LDA_Visualization6.html')


# Topic detection for Book 7


vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(H7_clean)

lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online',verbose=False)
data_lda = lda.fit_transform(data_vectorized)

# Keywords for topics clustered by Latent Dirichlet Allocation
print('##############THE 5 TOPICS FOR BOOK 7 ARE AS FOLLOWS###############')
print('\nLDA Model:')
selected_topics(lda, vectorizer)
# visualizing the results. An html interactive file will be created
dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(dash, 'LDA_Visualization7.html')
 
################################## END #######################################
