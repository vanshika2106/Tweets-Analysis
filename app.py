import io
from pydoc import describe
from click import progressbar
import streamlit as st
import pandas as pd
import numpy as np

# Data cleaning imports
import pandas as pd
import re
import numpy as np
import string
from nltk.tokenize import RegexpTokenizer

# WordCloud imports
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
import matplotlib.pyplot as plt

# Twitter API imports
import tweepy
import time
from keys import *

# FeatureSet imports
from textblob import TextBlob
from matplotlib.pyplot import pie, axis, show


# Model Generation imports
import nltk

import time

import twitapi


import warnings
warnings.filterwarnings('ignore')

############################## importing the project code

# code to scrape the tweets
import twitapi as tw
# code to preprocess the tweets
import preprocessing_cleaning_and_feature_set_updated as pre_clean_features

# code to generate the wordcloud
import wordcloud_generator as wc

# code to generate the feature set
import featureSetBuilder as fsb


# code to generate the model
import modelBuilder as mb

###############################################################################

_pass = 1
# form = st.form("my_form")
# form
# with st.form("my_form"):
#     st.write("Inside the form")
#     st_passcode = st.text_input("Enter your passcode")

#     # Every form must have a submit button.
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         if st_passcode == "XBEQH3ZTXOSJTCTZQW4T" or _pass == 1:
#             _pass = 1
#         else:
#             _pass = 0
#             st.write("Wrong passcode")
            

if _pass == 1:

    ################################################################################
    st.title('Twitter Keyword Sentiment Analysis')

    #Accpet the input from the user
    keyword = st.text_input("Enter the keyword to search for:")

    #select the filter option
    filter_option = st.selectbox("Select the filter option:", ["None", "Retweets", "Replies"])

    #select the number of tweets to be fetched
    tweet_count = st.slider("Select the number of tweets to be fetched:", 1, 100, 1)

    #start button to start the process
    start_button = st.button("Start")

    @st.cache(persist=True)
    def getData():
        df = pre_clean_features.setup()
        return df


    progressbar = st.progress(0)
    progressbar.text("TBD...")

    @st.cache(persist=True)
    def preProcess(df):
        df_pre_processed = pre_clean_features.preprocessing(df)
        return df_pre_processed

    @st.cache(persist=True)
    def cleanData(df_pre_processed):
        df_cleaned = pre_clean_features.cleaning(df_pre_processed)
        return df_cleaned

    @st.cache(persist=True)
    def getFeatureSet(df_cleaned):
        train_set, test_set = fsb.main(df_cleaned)
        return train_set, test_set


    @st.cache(persist=True)
    def getModel(train_set,test_set):
        classifier, accuracy = mb.main(train_set, test_set)
        return classifier, accuracy


    with st.container():    
        # step1 : Preprocess the training data
        st.title('Preprocessing Training Data')
        with st.spinner('Pre-processing Data...'):
            df = getData()
            df_pre_processed = preProcess(df)
            progressbar.progress(10)
            progressbar.text("Pre-processing Data...")
            st.success('Data Pre-processed!')
            buffer = io.StringIO()
            df_pre_processed.info(buf=buffer) 
            info = buffer.getvalue()
            st.text(info)
            st.dataframe(df_pre_processed.head(10))

    with st.container():
        # step2 : Clean the training data
        st.title('Cleaning  Data')
        with st.spinner('Cleaning Data...'):
            progressbar.text("Cleaning Data...")
            progressbar.progress(20)
            df_cleaned = cleanData(df_pre_processed)
            st.success('Data Cleaned!')
            buffer = io.StringIO()
            df_cleaned.info(buf=buffer) 
            info = buffer.getvalue()
            st.text(info)
            st.dataframe(df_cleaned.head(20))
    
    with st.container():
        # step3 : Generate the Word Cloud for the training data
        st.title('Generating Word Cloud')
        with st.spinner('Generating Word Cloud...'):
            print("Generating Word Cloud...")
            progressbar.text("Generating Word Cloud...")
            progressbar.progress(30)
            plt = wc.main(df_cleaned.clean_text)
            st.pyplot(plt)
            st.success('Word Cloud Generated!')
    

    with st.container():
        # step4 : Generate the Feature Set for the training data
        st.title('Generating Feature Set')
        with st.spinner('Generating features...'):
            print("Generating features...")
            train_set, test_set = getFeatureSet(df_cleaned)
            progressbar.text("Generating features...")
            progressbar.progress(40)
            st.write(train_set['Feature_Set'].astype(str))
            st.success('Features Generated!')


    
    with st.container():
        # step5 : Generate the Model for the training data
        st.title('Building Model')
        with st.spinner('Building Model...'):
            print("Building Model...")
            progressbar.text("Building Model...")
            progressbar.progress(50)
            classifier, accuracy = getModel(train_set, test_set)
            imp_feature = classifier.show_most_informative_features(10)
            st.text(imp_feature)
            st.write(accuracy)
            st.success('Model Built!')
    
    #submit button to start the process
    if start_button:
        with st.container():    
            # step6 : Scraping the tweets using twitter api
            st.title('Scraping Tweets')
            with st.spinner("Fetching tweets..."):
                twitter_df = tw.main(keyword, filter_option, tweet_count)
                progressbar.progress(60)
                progressbar.text("Fetching data...")
                st.success('Tweets Fetched!')

        with st.container():
            # step7 : Clean the twitter data
            st.title('Cleaning  twitter Data')
            with st.spinner('Cleaning twitter Data...'):
                progressbar.text("Cleaning Data...")
                progressbar.progress(70)
                twitter_df_cleaned = pre_clean_features.cleaning(twitter_df)
                st.success('Data Cleaned!')
                buffer = io.StringIO()
                twitter_df_cleaned.info(buf=buffer) 
                info = buffer.getvalue()
                st.text(info)
                st.dataframe(twitter_df_cleaned.head(20))

        with st.container():
            # step8 : Generate the word cloud for the scraped tweets
            st.title('Generating Word Cloud from Tweets')
            with st.spinner('Generating Word Cloud...'):
                print("Generating Word Cloud...")
                progressbar.text("Generating Word Cloud...")
                progressbar.progress(80)
                plt = wc.main(twitter_df_cleaned.clean_text)
                st.pyplot(plt)
                st.success('Word Cloud Generated!')

        with st.container():
            # step9 : Generate the Feature Set for the training data
            st.title('Generating Feature Set')
            with st.spinner('Generating features...'):
                print("Generating features...")
                twitter_df_cleaned = fsb.tweet_features(twitter_df_cleaned)
                progressbar.text("Generating features...")
                progressbar.progress(90)
                res = twitter_df_cleaned['Feature_Set'].astype(str)
                st.dataframe(res)
                st.success('Features Generated!')

        with st.container():
            # step10 : Run the classifier on the twitter data
            st.title('Running Model on twitter data')
            with st.spinner('Running Model on twitter data...'):
                print("Running Model on twitter data...")

                tweet_dfx = mb.tweet_classifier(classifier, twitter_df_cleaned)
                progressbar.text("Running Model on twitter data...")
                progressbar.progress(100)
                neg,pos = tweet_dfx.groupby('Pred_Sent').size()
                labels = []
                sizes = []
                for x, y in tweet_dfx.groupby('Pred_Sent').size().items():
                    labels.append(x)
                    sizes.append(y)
                    
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, labels=labels ,autopct='%.1f%%')
                ax1.axis('equal')
                ax1.set_title('Sentiment', fontsize=18)
                st.pyplot(fig1)

                st.success('Running Model on twitter data!')
        

