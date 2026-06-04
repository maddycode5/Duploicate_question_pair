import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re 
from bs4 import BeautifulSoup

import warnings 
warnings.filterwarnings("ignore")

df =pd.read_csv("train.csv")
new_df = df.sample(40000,random_state = 2)
print(new_df.head())

# PREPROCESSING THE DATA
def preprocess(q):
    q=str(q).lower().strip()

    # repllace the certain special chaarcters with their string equivalents 
    q= q.replace("%",'percent')
    q=q.replace('$','dollar')
    q=q.replace('₹','rupee')
    q=q.replace('ē','euro')
    q=q.replace('@','at')

    # the pattern math appear 900 times in whole dataset
    q=q.replace('[math]',"")

    # replacing the same number with strings equivalents 
    q=q.replace(',000,000,000 ', 'b')
    q=q.replace(',000,000 ', 'm')
    q=q.replace(',000 ','k')
    q=re.sub(r'([0-9]+)000000000',r'\1b',q)
    q=re.sub(r'([0-9]+)000000',r'\1m',q)
    q=re.sub(r'([0-9]+)000',r'\1k',q)

    # decontracting the words

    contractions = {
        "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
     "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q=' '.join(q_decontracted)
    q=q.replace("'ve","have")
    q=q.replace("n't","not")
    q=q.replace("'re","are")
    q=q.replace("'ll","will")

    # removing the HTML TAGS
    q=BeautifulSoup(q)
    q=q.get_text()

    # remove punctuations
    pattern = re.compile('\W')
    q=re.sub(pattern,' ',q).strip()

    return q

# now applying preprocessing step
new_df["question1"] =new_df["question1"].apply(preprocess)
new_df["question2"] = new_df["question2"].apply(preprocess)

print(new_df.head())

# FEATURE ENGINEERING
new_df['q1_len'] = new_df['question1'].str.len()
new_df['q2_len'] = new_df['question2'].str.len()

new_df["q1_num_words"] = new_df["question1"].apply(lambda row : len(row.split(" ")))
new_df["q2_num_words"] = new_df["question2"].apply(lambda row : len(row.split(" ")))

def common_words(row):
    w1 = set(map(lambda word : word.lower().strip(), row["question1"].split(" ")))
    w2 = set(map(lambda word : word.lower().strip(), row["question2"].split(" ")))
    return len(w1.intersection(w2))

new_df["word_common"] = new_df.apply(common_words , axis=1)

def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(),row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(),row["question2"].split(" ")) )
    return (len(w1)+len(w2))

new_df["word_total"] = new_df.apply(total_words ,axis =1)

new_df["words_share"] = round(new_df["word_common"]/new_df["word_total"],2)
print(new_df.head())

# ADVANCE FEATURES 
from nltk.corpus import stopwords
def fetch_token_features(row):
    q1=row['question1']
    q2=row['question2']

    SAFE_DIV = 0.0001
    stop_words =stopwords.words("english")
    token_features =[0.0]*8

    # converting the sentence into tokens
    q1_token =  q1.split()
    q2_token = q2.split()

    if len(q1_token) == 0 or len(q2_token) ==0:
        return token_features
    
    # get the non stopwords in questions

    q1_words = set([word for word in q1_token if word not in stop_words])
    q2_words = set([word for word in  q2_token if word not in stop_words])

    # get the stop words from the question pair 
    q1_stop = set([word for word in q1_token if word in stop_words ])
    q2_stop = set([word for word in q2_token  if word in stop_words])

    # get the common non stopwords from question pair    
    common_word_count = len(q1_words.intersection(q2_words))

    # get the common non stopwords from question pair 
    common_stop_count = len(q1_words.intersection(q2_words))

    #  get the common tokens from question pair
    common_token_count = len(set(q1_token).intersection(set(q2_token)))

    token_features[0] = common_word_count / (min(len(q1_words),len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words),len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stop),len(q2_stop)) + SAFE_DIV)
    token_features[3] = common_stop_count / (min(len(q1_stop),len(q2_stop)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_token),len(q2_token)) + SAFE_DIV)
    token_features[5] = common_token_count / (min(len(q1_token),len(q2_token)) + SAFE_DIV)

    # last word is same or not 
    token_features[6] = int(q1_token[-1] == q2_token[-1])

    # first word is same or not 
    token_features[7] = int (q1_token[0]== q2_token[0])

    return token_features
