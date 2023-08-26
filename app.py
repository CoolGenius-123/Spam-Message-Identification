import streamlit as st
import pandas as pd
import pickle
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()

# writing the function for text preprocessing
def text_preprocess(text):
    
    # converting the text to lower case
    text = text.lower()
    
    # tokenizing the text
    text = nltk.word_tokenize(text)
    
    # Removing punctuation only keeping alpha numeric characters
    
    text = [word for word in text if word.isalnum()]
    
    # Removing stop words
    
    text = [word for word in text if word not in stopwords.words('english')]
    
    # stemming words
    
    text = [ps.stem(word) for word in text]
    
    # Convert list of tokens back to string
    
    text = ' '.join(text)
    
    
    return text

# loading the trained model
pickle_in = open('model.pkl', 'rb')

# loading the vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# creating heading for the app
st.title("Email Spam Classifier")

# creating the text box for user input
text = st.text_input("Enter the text/message to be classified as spam or not spam")

# creating the button for prediction
btn = st.button("Predict")

# checking if the button is pressed or not
if btn and text:
    # loading the model
    model = pickle.load(pickle_in)
    # preprocessing the text
    text = text_preprocess(text)
    # vectorizing the user input
    vect_text = vectorizer.transform([text]).toarray()
    # predicting the output
    result = model.predict(vect_text)
    # displaying the result
    if result[0] == 1:
        st.error("This is a Spam Email")
    else:
        st.success("This is a Ham Email")

# how to run the app
# streamlit run app.py
