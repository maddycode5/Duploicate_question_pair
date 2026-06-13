import streamlit as st
import helper

import pickle

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))
 
st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = helper.query_point_creator(q1,q2)

    st.write("Prediction:", model.predict(query))
    st.write("Probability:", model.predict_proba(query))

    result = model.predict(query)[0]

    if result == 1:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')

import os
import time

print(time.ctime(os.path.getmtime('model.pkl')))