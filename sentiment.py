import numpy as np
import pandas as pd
import re
import emoji
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score,f1_score
import streamlit as st
import pickle
result = None

st.title("Sentiment Analysis on FlipKart Reviews")
text = st.text_input("Enter the Review")
import os

current_dir = os.path.dirname(__file__)

pickle_file_path = os.path.join(current_dir, "C:\\Users\\Rahul\\Desktop\\Internship\\TASK 8\\webapp\\webapp\\sentiment_yonex.pkl")

with open(pickle_file_path, "rb") as f:
    model = pickle.load(f)
if st.button("Submit")==True:
    result = model.predict([text])[0]

if result == 'Positive':
    st.caption(':green[Positive]')
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS_GMI9aISd0YG6JeHDkEuPXRSKsEaKOw9n7A&usqp=CAU")
elif result == 'Negative':
    st.caption(':red[Negative]')
    st.image("https://img.freepik.com/premium-photo/man-no-money-close-his-face-feel-sadness-with-no-money_483229-3294.jpg")