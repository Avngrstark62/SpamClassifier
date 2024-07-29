import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import streamlit as st

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

cv = pickle.load(open('pickle_files/count_vectorizer.pkl', 'rb'))

model = pickle.load(open('pickle_files/spam_model.pkl', 'rb'))


def spam_or_ham(message):
    message = re.sub('[^a-zA-Z]', ' ', message)
    message = message.lower()
    message = message.split()
    message = [lemmatizer.lemmatize(word) for word in message if word not in set(stopwords.words('english'))]
    message = ' '.join(message)
    
    X = cv.transform([message]).toarray()
    prediction = model.predict(X)
    if prediction:
        return 'Not Spam'
    else:
        return 'Spam'


st.title("Spam Classifier")

message = st.text_input("Type a Message")

if st.button("Check Spam or Ham"):
    if message:
        spam_check = spam_or_ham(message)
        st.write(spam_check)
    else:
        st.write('Empty Message')