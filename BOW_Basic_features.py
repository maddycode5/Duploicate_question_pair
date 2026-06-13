import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 

df =pd.read_csv("train.csv")
# print(df.shape)

# print(df.head())
# print(df.info()) #info about the dataset

# print(df.sample(10))  #random samples taken from the dataset

# # missing values
# print(df.isnull().sum())

# duplicate values
# print(df.duplicated().sum())

# distribution of duplicate and non-duplicate questions
# print(df['is_duplicate'].value_counts())
# print((df["is_duplicate"].value_counts()/df["is_duplicate"].count())*100)
# df["is_duplicate"].value_counts().plot(kind='bar')
# plt.xlabel('Is Duplicate')
# plt.ylabel('Count')
# plt.title('Distribution of Duplicate and Non-Duplicate Questions')
# plt.show()


# Now simplest way to check the dataset by applying BOW (Bag of words ) on question 1 and 2 snf then applying random forest like algorithm 

new_df = df.sample(30000) #taking 30000 samples from datasets 
# print(new_df.isnull().sum())

# print(new_df.duplicated().sum())

# new dataset taken where only question are considered and then applying BOW on it
ques_df = new_df[['question1','question2']]
# print(ques_df.head())

# Now BOW (Bag of words ) on question 1 and 2
from sklearn.feature_extraction.text import CountVectorizer
# merge texts
questions = list(new_df["question1"]) + list(ques_df["question2"])

cv =CountVectorizer(max_features = 4000 )
q1_arr ,q2_arr = np.split(cv.fit_transform(questions).toarray(),2)
print(q1_arr.shape)
print(q2_arr.shape)

# BOW Feature Extraction is done and we have two arrays for question 1 and question 2. Now we will concatenate these two arrays to create a single feature set for each question pair.    
# concatenating the two arrays to create a single feature set for each question pair
temp_df1 = pd.DataFrame(q1_arr,index =ques_df.index)
temp_df2 = pd.DataFrame(q2_arr,index =ques_df.index)
temp_df =  pd.concat([temp_df1,temp_df2], axis =1)
print(temp_df.head) # to check the concatenated dataframe

# new features is duplicate or not is added to the temp_df from original dataset 
temp_df["is_duplicate"] = new_df["is_duplicate"]
print(temp_df.head())

# applying train_test_split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(temp_df.iloc[:,0:-1].values,temp_df.iloc[:,-1].values,test_size = 0.2,random_state=42)

# # now applying Random Forest Classifier on the temp_df to predict whether the question pairs are duplicate or not
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# rf = RandomForestClassifier()
# rf.fit(X_train , y_train)
# y_pred =rf.predict(X_test)

# print("Accuracy of the model is: ", accuracy_score(y_test,y_pred))

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
xgb = XGBClassifier()
xgb.fit(X_train , y_train)
y_pred =xgb.predict(X_test)
print("Accuracy of the model is: ", accuracy_score(y_test,y_pred))