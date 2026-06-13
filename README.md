# Duplicate Question Pair Detection

A machine learning-based web application that predicts whether two questions are duplicates using NLP feature engineering, TF-IDF vectorization, and XGBoost classification.

### Tech Stack

* Python
* Scikit-learn
* XGBoost
* NLTK
* Streamlit

### Features

* Text preprocessing and cleaning
* 22 handcrafted NLP similarity features
* TF-IDF vectorization (5000 features)
* Duplicate question prediction
* Interactive Streamlit interface

### Model Performance

* Random Forest Accuracy: **78.6%**
* XGBoost Accuracy: **79.6%**

### Run Locally

```bash
python advance_features.py
streamlit run streamlit-app/app.py
```
