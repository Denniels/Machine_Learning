
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import  pandas as pd
import re

import string
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')


def vectorize_sentiment(df, sentimiento = 'sentiment', objetivo = 'content', stop_words_ = 'english', plot = False):
    
    """
    Devuelve una matriz dispersa para analisis de texto.
    """
    
    df_tmp = df.copy()
    count_vectorizer = CountVectorizer(stop_words = stop_words_, max_features = 100)
    count_vectorizer_fit = count_vectorizer.fit_transform(df_tmp[df_tmp['sentiment'] == sentimiento][objetivo])
    
    words = count_vectorizer.get_feature_names()
    words_freq = count_vectorizer_fit.toarray().sum(axis = 0)
    
    df_words = pd.DataFrame(zip(words, words_freq), columns = ['word', 'freq']).sort_values(by = 'freq', ascending = False)
    
    if plot == True:
        plt.figure(figsize = (25,7)) # No funciona ahi.
        sns.barplot(x = df_words['word'], y = df_words['freq']) 
        plt.title(sentimiento)
        plt.xticks(rotation = 90)
    
    return df_words


def clean_text(df, col_name, new_col_name):
    """
    Funcion que limpia el dataset de caracteres especiales como A-Za-z0-9 etc.
    """
    # column values to lower case
    df[new_col_name] = df[col_name].str.lower().str.strip()
    # removes special characters
    df[new_col_name] = df[new_col_name].apply(lambda x: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z.% \t])", "", x))
    return df

def replace_stop_words(df, col, stop_list):
    df['{}_stop'.format(col)] = df[col].apply(lambda x: ' '.join([word for word in x.split() if x not in stop_list]))
    return df

def word_lemmatizer(text):
    text_lemma = [WordNetLemmatizer().lemmatize(word) for word in text]
    return text_lemma

stop_words = stopwords.words('english')
def nlp_cleaning(df):
    """
    Devuelve un dataset con el preprocesamiento listo.
    """
    # normalization
    df = clean_text(df, 'content', 'content_norm')
    # remove stop words
    df = replace_stop_words(df, 'content_norm', stop_words)
    # removing numbers
    df['content_norm_stop'] = df['content_norm_stop'].apply(lambda x: re.sub('[0-9]+', '', x))
    # tokenize text
    df['content_token'] = df['content_norm_stop'].apply(lambda x: word_tokenize(x))
    # lemmatization
    df['content_token_lemma'] = df['content_token'].apply(lambda x: word_lemmatizer(x))
    # joining lemmas and removing punctuation
    df['content_clean'] = df['content_token_lemma'].apply(lambda list_: ' '.join([word for word in list_ if word not in string.punctuation]))
    return df