import pickle
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "streamlit-app"))

import helper
model = pickle.load(open('model.pkl', 'rb'))

pairs = [
    ("What is machine learning?", "What is machine learning?"),
    ("What is your name?", "How many countries are there in the world?"),
    ("How can I learn Python?", "What is the best way to learn Python?")
]

for q1, q2 in pairs:
    query = helper.query_point_creator(q1, q2)

    print("\nQ1:", q1)
    print("Q2:", q2)
    print("Prediction:", model.predict(query))
    print("Probability:", model.predict_proba(query))

