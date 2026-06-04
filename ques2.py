# FEATURES ADDED IN THIS CODE  AND FEATURE ENGINEERING DONE

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 

df =pd.read_csv("train.csv")
# print(df.shape)

# print(df.head())
# print(df.info()) #info about the dataset

new_df = df.sample(30000,random_state=2)
# print(new_df)

# # missing values
# print(df.isnull().sum())

# # duplicate values
# print(df.duplicated().sum())

# # distribution of duplicate and non-duplicate questions
# print(new_df['is_duplicate'].value_counts())
# print((new_df["is_duplicate"].value_counts()/new_df["is_duplicate"].count())*100)
# new_df["is_duplicate"].value_counts().plot(kind='bar')
# plt.xlabel('Is Duplicate')
# plt.ylabel('Count')
# plt.title('Distribution of Duplicate and Non-Duplicate Questions')
# plt.show()


# # now repeated questions in question1 and question2
# qid = pd.Series(new_df['qid1'].tolist() + new_df['qid2'].tolist())
# print("Number of unique questions: ", np.unique(qid).shape[0])
# x = qid.value_counts()>1
# print("Number of repeated questions: ", x[x].shape[0])


# # repeated question histogram
# plt.hist(qid.value_counts(),bins =160)
# plt.yscale('log')
# plt.xlabel('Number of times question is repeated')
# plt.ylabel('Number of questions')
# plt.title('Histogram of Repeated Questions')
# plt.show()

# FEATURE ENGINEERING
# first feature is the length of question 1 and question 2
new_df["q1_len"] = new_df["question1"].str.len()
new_df['q2_len'] = new_df["question2"].str.len()
print(new_df.head())

# 2nd Feature number of words in question 1 and question 2
new_df['q1_num_words'] = new_df['question1'].apply(lambda row : len(row.split(" ")))
new_df["q2_num_words"] = new_df['question2'].apply(lambda row : len(row.split(" ")))
print(new_df.head())

# 3rd feature is the number of common words in question 1 and question 2
def common_words(row):
    w1=set(map(lambda word :word.lower().strip(), row['question1'].split(" ")))
    w2=set(map(lambda word :word.lower().strip(), row['question2'].split(" ")))
    return len(w1.intersection(w2))

new_df['words_common'] = new_df.apply(common_words , axis =1)
print(new_df.head())

# 4th features is the total number of unique words in question 1 and question 2
def total_unique_words(row):
    w1=set(map(lambda word :word.lower().strip(), row['question1'].split(" ")))
    w2=set(map(lambda word :word.lower().strip(), row['question2'].split(" ")))
    return (len(w1)  + len(w2))

new_df['word_total'] = new_df.apply(total_unique_words ,axis =1)
print(new_df.head())

# 5th feature is word share which is the ratio of common words to total unique words in question 1 and question 2
new_df['words_share'] = round(new_df['words_common'] / new_df['word_total'],2)
print(new_df.head())


# NOW NEW FRAME IS READY WITH NEW FEATURES AND FEATURE ENGINEERING DONE.

ques_df = new_df[["question1" , "question2"]]
print(ques_df.head())

# NOW DROPPING FEATURES FROM ORIGINAL DATAFRAME
# id, qid1,qid2 ,question 1 question2 
final_df = new_df.drop(columns=["id" ,"qid1" ,"qid2", "question1","question2"])
print(final_df.shape)
print(final_df.head())

# IS_DUPLICATE is the output feature and rest are input features 

# NOW WE CAN APPLY BOW (Bag of words ) ON QUESTION 1 AND QUESTION 2
#  AND THEN APPLYING RANDOM FOREST LIKE ALGORITHM TO CHECK THE DUPLICATE QUESTIONS.

from sklearn.feature_extraction.text import CountVectorizer
# merge texts
questions = list(ques_df["question1"]) + list(ques_df["question2"])

cv =CountVectorizer(max_features=3000)
q1_arr ,q2_arrr = np.vsplit(cv.fit_transform(questions).toarray(),2)

temp_df1 = pd.DataFrame(q1_arr ,index=ques_df.index)
temp_df2 = pd.DataFrame(q2_arrr ,index=ques_df.index)
temp_df = pd.concat([temp_df1,temp_df2],axis=1)
# print(temp_df.head())
print(temp_df.shape) #here (30000,6000) because we have 3000 features for question 1 and 3000 features for question 2

# NOW CONCATENATING THE NEW FEATURES WITH THE BOW FEATURES
final_df = pd.concat([final_df,temp_df],axis=1)
print(final_df.shape) #here (30000, 6008) because we have 8 features and 6000 BOW features
print(final_df.head())
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(final_df.iloc[:,1:].values,final_df.iloc[:,0].values,test_size=0.2,random_state=2)


# APPLYING RANDOM FOREST CLASSIFIER TO CHECK THE DUPLICATE QUESTIONS
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
rf =RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred =rf.predict(X_test)
# print("Accuracy score of the model is : " ,accuracy_score(y_test,y_pred))

from xgboost import XGBClassifier 
xgb =XGBClassifier()
xgb.fit(X_train,y_train)
y_pred =xgb.predict(X_test)
print("Accuracy score of the model is : " ,accuracy_score(y_test,y_pred))
