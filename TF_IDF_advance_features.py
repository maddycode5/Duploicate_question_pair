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
# print(new_df.head())

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

# print(new_df.head())

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
# print(new_df.head())

# ADVANCE FEATURES 
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def fetch_token_features(row):
    q1=row['question1']
    q2=row['question2']

    SAFE_DIV = 0.0001
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
    common_stop_count = len(q1_stop.intersection(q2_stop))

    #  get the common tokens from question pair
    common_token_count = len(set(q1_token).intersection(set(q2_token)))

    token_features[0] = common_word_count / (min(len(q1_words),len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words),len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stop),len(q2_stop)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stop),len(q2_stop)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_token),len(q2_token)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_token),len(q2_token)) + SAFE_DIV)

    # last word is same or not 
    token_features[6] = int(q1_token[-1] == q2_token[-1])

    # first word is same or not 
    token_features[7] = int (q1_token[0]== q2_token[0])

    return token_features

token_features = new_df.apply(fetch_token_features , axis=1)

new_df["cwc_min"] = token_features.apply(lambda x: x[0])
new_df["cwc_max"] = token_features.apply(lambda x: x[1])
new_df["csc_min"] = token_features.apply(lambda x: x[2])
new_df["csc_max"] = token_features.apply(lambda x: x[3])
new_df["ctc_min"] = token_features.apply(lambda x: x[4])
new_df["ctc_max"] = token_features.apply(lambda x: x[5])
new_df["last_word_eq"] = token_features.apply(lambda x: x[6])
new_df["first_word_eq"] = token_features.apply(lambda x: x[7])
# print(new_df.head())


import distance 
def fetch_length_features(row):
    q1= row['question1']
    q2=row['question2']

    length_features = [0.0]*3

    # converting the sentence into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    # absolute length features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens)) 

    # average token length of both questions
    length_features[1]= (len(q1_tokens) +len(q2_tokens)) /2

    strs =list(distance.lcsubstrings(q1,q2))
    length_features[2] = len(strs[0]) / (min(len(q1),len(q2)) + 1)

    return length_features

length_features = new_df.apply(fetch_length_features ,axis=1)

new_df["abs_len_diff"] = list(map(lambda x :x[0] , length_features))
new_df["mean_len"] = list(map (lambda x : x[1] ,length_features))
new_df["longest_substr_ratio"] = list(map(lambda x: x[2],length_features))

# print(new_df.head())

# # FUZZY FEATURES
from fuzzywuzzy import fuzz

def fetch_fuzzy_features(row):
    q1 = row["question1"]
    q2 = row['question2']

    fuzzy_features = [0.0]*4

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1,q2)

    # fuzzy_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1,q2)

    # fuzz_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1,q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1,q2)

    return fuzzy_features

fuzz_features = new_df.apply(fetch_fuzzy_features ,axis =1)

# creating new features columns for fuzzy features
new_df["fuzz_ratio"] = list(map(lambda x :x[0] ,fuzz_features))
new_df["fuzz_partial_ratio"] = list(map(lambda x : x[1] , fuzz_features))
new_df["token_sort_ratio"] = list(map(lambda x : x[2] , fuzz_features)) 
new_df["token_set_ratio"] = list(map(lambda x : x[3] , fuzz_features))


# print(new_df.head())

sns.pairplot(new_df[["ctc_min","cwc_min","csc_min","is_duplicate"]],hue="is_duplicate")
# plt.show()

sns.pairplot(new_df[["ctc_max","cwc_max","csc_max","is_duplicate"]],hue="is_duplicate")
# plt.show()

sns.pairplot(new_df[["last_word_eq" , "first_word_eq" , "is_duplicate"]] , hue= "is_duplicate")
# plt.show()

sns.pairplot(new_df[["mean_len" , "abs_len_diff","longest_substr_ratio" , "is_duplicate"]],hue= "is_duplicate")
# plt.show()

sns.pairplot(new_df[["fuzz_ratio","fuzz_partial_ratio","token_sort_ratio","token_set_ratio","is_duplicate"]],hue = "is_duplicate")
# plt.show()

# using TSNE for dimensionality reduction for 15 features (generated afetr cleaming the data ) to 3 dimension

from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(new_df[['cwc_min','cwc_max','csc_min','csc_max','ctc_min','ctc_max','last_word_eq',"first_word_eq","mean_len" , "abs_len_diff","longest_substr_ratio" , "fuzz_ratio","fuzz_partial_ratio","token_sort_ratio","token_set_ratio"]])
y = new_df['is_duplicate'].values
# print(X)  
# print(y)

from sklearn.manifold import TSNE

tsne2d =TSNE(
    n_components = 2,
    init = 'random' , #pca
    random_state = 101,
    method = "barnes_hut",
    max_iter = 1000,
    verbose = 2,
    angle = 0.5
).fit_transform(X)

x_df = pd.DataFrame({'x':tsne2d[:,0], 'y':tsne2d[:,1] ,'label':y})

tsne3d = TSNE(    n_components = 3,
    init = 'random' , #pca
    random_state = 101,
    method = "barnes_hut",
    max_iter = 1000,
    verbose = 2,
    angle = 0.5
).fit_transform(X)


ques_df = new_df[['question1','question2']]

final_df = new_df.drop(columns=['id','qid1','qid2','question1','question2'])
# print(final_df.shape)
final_df.head()


# Applying TF_IDF 
from sklearn.feature_extraction.text import TfidfVectorizer

questions = pd.concat([new_df['question1'], new_df['question2']]).unique()

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

tfidf.fit(questions)

q1_tfidf = tfidf.transform(new_df['question1'])
q2_tfidf = tfidf.transform(new_df['question2'])

# print(q1_tfidf.shape)
# print(q2_tfidf.shape)
# print(final_df.shape)


from scipy.sparse import csr_matrix, hstack

print(final_df.columns.tolist())
X_features = csr_matrix(final_df.iloc[:,1:].astype('float32').values)


X = hstack([
    X_features,
    q1_tfidf,
    q2_tfidf
])
y = final_df.iloc[:,0].values

print(X.shape)

# from sklearn.feature_extraction.text import CountVectorizer
# # merge texts
# questions = list(ques_df['question1']) + list(ques_df['question2'])

# cv = CountVectorizer(max_features=3000)
# q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(),2)


# temp_df1 = pd.DataFrame(q1_arr, index= ques_df.index)
# temp_df2 = pd.DataFrame(q2_arr, index= ques_df.index)
# temp_df = pd.concat([temp_df1, temp_df2], axis=1)
# temp_df.shape


# final_df = pd.concat([final_df, temp_df], axis=1)
# # print(final_df.shape)
# # final_df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print("The Random Forest accuracy_score is : ", accuracy_score(y_test,y_pred))

# XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

xgb.fit(X_train, y_train)
y_pred1 = xgb.predict(X_test)
print("The XGBoost accuracy_score is : " , accuracy_score(y_test,y_pred1))

# CONFUSION MATRIX
from sklearn.metrics import confusion_matrix

# for random forest model
print("The confusion matrix of RandomForest Model is: ", confusion_matrix(y_test,y_pred))

# for xgboost model
print("The confusion matrix of XGBoost Model is: ", confusion_matrix(y_test,y_pred1))

def test_common_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return len(w1 & w2)

def test_total_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return (len(w1) + len(w2))


STOP_WORDS = set(stopwords.words("english"))
def test_fetch_token_features(q1,q2):
    
    SAFE_DIV = 0.0001     
    token_features = [0.0]*8
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features

def test_fetch_length_features(q1,q2):
    
    length_features = [0.0]*3
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens))/2
    
    strs = list(distance.lcsubstrings(q1, q2))

    if len(strs) > 0:
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    else:
        length_features[2] = 0
    return length_features

def test_fetch_fuzzy_features(q1, q2):

    fuzzy_features = [0.0] * 4

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features

def query_point_creator(q1,q2):
    
    input_query = []
    
    # preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    
    # fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))
    
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    
    common = test_common_words(q1, q2)
    total = test_total_words(q1, q2)

    input_query.append(common)
    input_query.append(total)

    if total == 0:
        input_query.append(0)
    else:
        input_query.append(round(common / total, 2))

    # fetch token features
    token_features = test_fetch_token_features(q1,q2)
    input_query.extend(token_features)
    
    # fetch length based features
    length_features = test_fetch_length_features(q1,q2)
    input_query.extend(length_features)
    
    # fetch fuzzy features
    fuzzy_features = test_fetch_fuzzy_features(q1,q2)
    input_query.extend(fuzzy_features)
    
    # tfidf feature for q1
    q1_tfidf1 = tfidf.transform([q1]).toarray()
    
    # tfidf feature for q2
    q2_tfidf2 = tfidf.transform([q2]).toarray()

    print("Q1:", q1)
    print("Q2:", q2)

    print("Basic + Advanced Features:")
    print(input_query)

    print("Token Features:", token_features)
    print("Length Features:", length_features)
    print("Fuzzy Features:", fuzzy_features)
    return np.hstack((np.array(input_query).reshape(1,-1),q1_tfidf1 ,q2_tfidf2))

query = query_point_creator(
    "What is your name?",
    "How many countries are there in the world?"
)

prediction = xgb.predict(query)

if prediction[0] == 1:
    print("Duplicate Questions")
else:
    print("Not Duplicate Questions") 

q1 = preprocess("What is your name?")
q2 = preprocess("How many countries are there in the world?")

print(test_fetch_token_features(q1,q2))
print(test_fetch_length_features(q1,q2))
print(test_fetch_fuzzy_features(q1,q2))


import pickle
pickle.dump(xgb,open('model.pkl','wb'))
pickle.dump(tfidf,open("tfidf.pkl",'wb'))

