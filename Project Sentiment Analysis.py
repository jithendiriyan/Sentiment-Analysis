#!/usr/bin/env python
# coding: utf-8

# In[58]:


#Importing the libraries 
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[47]:


#importing the data as dataset
dataset = pd.read_csv("E:\DS project\Project Sentiment Analysis\IMDB Dataset.csv")
dataset


# In[48]:


#processing the step by step
#step 1:EDA
dataset.isnull().sum() # prints if any null values is present or not in dataset


# In[49]:


dataset.describe() # Returns description of the data in the Dataset


# In[59]:


dataset.head()


# In[62]:


text_data = dataset['text']


# In[60]:


stopwords.fileids()

stop = stopwords.words('english')

s = dataset
word = word_tokenize(s)

filtered_list = []
for w in word:
    if w not in stop:
        filtered_list.append(w)
       
filtered_list 


# In[ ]:




