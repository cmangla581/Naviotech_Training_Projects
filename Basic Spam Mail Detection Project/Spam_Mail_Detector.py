
# Here, we will be making a Streamlit UI for the Spam Mail Detection Project, so that it can be easily used by the users 

import streamlit as st 
import numpy as np 
import pandas as pd 
import string 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS 
from sklearn.naive_bayes  import MultinomialNB 
from sklearn.metrics import accuracy_score, confusion_matrix  

# Setting the page confisuration 
st.set_page_config(page_title = "Spam Mail Detector", layout = "centered") 

# Now, we will be putting the custom css 
st.markdown("""
<style>

/* MAIN BACKGROUND */
body, .stApp {
    background-color: #000000;
    color: white;
}

/* MAIN TITLE GLOW */
h1 {
    color: #00FFFF;
    text-align: center;
    font-weight: bold;
    text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF, 0 0 40px #00FFFF;
}

/* SUBHEADINGS GLOW */
h2, h3 {
    color: #FFD700;
    text-shadow: 0 0 8px #FFD700, 0 0 20px #FFD700;
}

/* INPUT BOX */
.stTextInput input {
    background-color: #111111;
    color: white;
    border-radius: 10px;
    border: 1px solid #00FFFF;
}

/* BUTTON STYLE */
.stButton button {
    background: linear-gradient(45deg, #00FFFF, #00FF7F);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 25px;
    border: none;
    box-shadow: 0 0 10px #00FFFF;
}

/* ACCURACY CARD */
.accuracy-box {
    background: #111111;
    border: 2px solid #00FF7F;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: #00FF7F;
    text-shadow: 0 0 10px #00FF7F;
    box-shadow: 0 0 25px #00FF7F;
}

/* DATAFRAME TEXT */
.dataframe {
    color: white;
} 
            
/* Result box base style */
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
    animation: glow 1.5s infinite alternate;
}

/* Make ALL labels visible */
label {
    color: #00ffff !important;
    font-size: 20px !important;
    font-weight: bold !important;
    text-shadow: 0 0 10px #00ffff;
}

/* Input box styling */
textarea {
    background-color: #111 !important;
    color: white !important;
    font-size: 16px !important;
    border: 2px solid #00ffff !important;
    caret-color: #00ffff !important;   /* Cursor color */
}

/* Glow effect when typing */
textarea:focus {
    border: 2px solid #00ffff !important;
    box-shadow: 0 0 20px #00ffff !important;
    outline: none !important;
} 
            
textarea, input {
    caret-color: #00ffff !important;   /* Bright neon cursor */
}
                       

/* Spam style */
.spam {
    background: rgba(255, 0, 0, 0.15);
    color: #ff4d4d;
    border: 2px solid #ff4d4d;
    box-shadow: 0 0 20px #ff4d4d;
}

/* Not spam style */
.ham {
    background: rgba(0, 255, 120, 0.15);
    color: #00ff99;
    border: 2px solid #00ff99;
    box-shadow: 0 0 20px #00ff99;
}

/* Glow animation */
@keyframes glow {
    from { box-shadow: 0 0 10px currentColor; }
    to { box-shadow: 0 0 30px currentColor; }
}

</style>
""", unsafe_allow_html=True)


st.title("📧 Spam Mail Detection System") 
st.write("Machine Leaarning based Spam vs Ham Classifier")  

# Now, we will be loading the dataset for the further analysis 
data = pd.read_csv("mail_data.csv")  

# Now , we will be converting the categorical to the numefical data 
le = LabelEncoder() 
for col in ["Category"]: 
    data[col] = le.fit_transform(data[col]) 

# Noow, we will be doing the preprocessing of the texxt as  

def preprocess_text(text): 
    text = text.lower() 

    text = "".join([char for char in text if char not in string.punctuation])  
    words = text.split() 
    words = [word for word in words if word not in ENGLISH_STOP_WORDS] 
    return " ".join(words) 

data['clean_message'] = data['Message'].apply(preprocess_text)  

# ------------ TFIDF -------------- 
tfidf = TfidfVectorizer(max_features = 3000) 
x = tfidf.fit_transform(data['clean_message']).toarray() 
y = data.filter(['Category'], axis = 1)  

# Training the model  
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.2)  

model = MultinomialNB() 
a = model.fit(x_train, y_train) 

y_pred = a.predict(x_test) 
 
accuracy = accuracy_score(y_test, y_pred)  
cm = confusion_matrix(y_test, y_pred)

# Now, we will be putting the dataset preview  

st.subheader("📊 Dataset Preview") 
st.dataframe(data.head()) 

# Accuarcy Display to show in the project 

st.subheader("🎯 Model Accuracy") 
st.markdown(f"""
<div class="accuracy-box">
    Accuracy Score: {accuracy*100:.2f}%
</div>
""", unsafe_allow_html=True)   

# Plotting of the Confusion Matrix  
st.subheader("📉 Confusion Matrix") 

fig, ax = plt.subplots()  
sns.heatmap(cm, annot = True, fmt = "d", cmap = "coolwarm",
            xticklabels = ["Ham", "Spam"],
            yticklabels = ["Ham", "Spam"])  
plt.xlabel("Predicted") 
plt.ylabel("Actual")  

st.pyplot(fig) 

# Now, comes the prediction section, where we will be making the predictions on the dataset 

st.subheader("✉️ Check Your Message") 

user_input = st.text_input("Enter Email / SMS Text") 

def predict_spam(message): 
    message = preprocess_text(message) 
    vector = tfidf.transform([message]).toarray() 
    pred = a.predict(vector) 

    return "🚨 SPAM MESSAGE" if pred[0] == 1 else "✅ NOT SPAM"

if st.button('Predict'):

    vectorized = tfidf.transform([user_input]) 
    prediction = a.predict(vectorized)[0] 

    if prediction == 1:
        st.markdown(
            '<div class="result-box spam">🚨 SPAM MESSAGE DETECTED</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box ham">✅ SAFE MESSAGE (NOT SPAM)</div>',
            unsafe_allow_html=True
        ) 

st.markdown("--------------") 
st.caption("Built by Chaitanya Mangla using Streamlit and Naive Bayes") 
