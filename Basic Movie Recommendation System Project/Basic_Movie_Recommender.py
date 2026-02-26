# Here, in this project, we will be making a stramlit predictor app for the Basic Movie Recommendation System, whose analysis
# we have done on the Jupyter Notebook 
import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns  
import difflib 

from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity 

# Now, we will be setting the ppage configuration  
st.set_page_config(page_title = "Movie Recommender", layout = "centered") 

# Now, we will be importing the css via streamlit markdow, to give the frontend a beautiful look. 

st.markdown("""
<style>

/* =====================================
   🌈 MOVING ANIMATED BACKGROUND
===================================== */

.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c3c);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
    color: white;
}

/* Background animation */
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}


/* =====================================
   👑 ANIMATED GLOWING MAIN TITLE
===================================== */

h1 {
    text-align: center;
    font-size: 75px !important;
    font-weight: 900 !important;
    letter-spacing: 3px;
    color: #ffffff !important;

    animation: glowPulse 2s ease-in-out infinite alternate;

    text-shadow:
        0 0 10px #00eaff,
        0 0 20px #00eaff,
        0 0 40px #0072ff,
        0 0 60px #0072ff,
        0 0 80px #00c6ff;
}

/* Glowing animation */
@keyframes glowPulse {
    from {
        text-shadow:
            0 0 10px #00eaff,
            0 0 20px #00eaff,
            0 0 40px #0072ff;
    }
    to {
        text-shadow:
            0 0 20px #00eaff,
            0 0 40px #00eaff,
            0 0 60px #0072ff,
            0 0 100px #00c6ff;
    }
}


/* =====================================
   ✨ SUBHEADINGS
===================================== */

h2, h3 {
    color: white !important;
    font-weight: 700;
    text-shadow:
        0 0 8px #00c3ff,
        0 0 18px #0072ff;
}


/* =====================================
   🏷 INPUT LABEL
===================================== */

label {
    color: white !important;
    font-weight: 600 !important;
    font-size: 17px !important;
    text-shadow: 0 0 6px #00c3ff;
}


/* =====================================
   ✍ INPUT BOX
===================================== */

.stTextInput > div > div > input {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 12px !important;
    border: 2px solid #00c3ff !important;
    padding: 10px !important;
    font-size: 16px !important;
}

.stTextInput input {
    caret-color: #00eaff !important;
}


/* =====================================
   🔘 BUTTON
===================================== */

div.stButton > button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white !important;
    font-weight: bold;
    border-radius: 30px;
    padding: 10px 25px;
    border: none;
    transition: 0.3s ease-in-out;
}

div.stButton > button:hover {
    transform: scale(1.1);
    box-shadow: 0 0 15px #00eaff;
}


/* =====================================
   🎬 MOVIE LIST (SHINY BLUE)
===================================== */

.stMarkdown p {
    color: #a0e9ff !important;
    font-weight: 700;
    font-size: 18px;

    text-shadow:
        0 0 6px #00c3ff,
        0 0 12px #0072ff,
        0 0 20px #00eaff;
}


/* =====================================
   ❌ DECORATIVE ERROR MESSAGE
===================================== */

div[data-testid="stAlert"] {
    background: linear-gradient(45deg, #2c003e, #4a0072);
    color: #ffb3ff !important;
    font-weight: bold;
    font-size: 18px;
    border-radius: 12px;
    text-align: center;

    box-shadow:
        0 0 10px #ff00ff,
        0 0 20px #cc00ff;
}


/* =====================================
   📊 DATAFRAME
===================================== */

[data-testid="stDataFrame"] {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px;
}


/* =====================================
   📈 CHART AREA
===================================== */

canvas {
    background-color: #1e293b !important;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
# Setting the title of the recommender app as: 

st.markdown("<h1>🎬 Basic Movie Recommendation System</h1>", unsafe_allow_html=True) 

# Now, we will be loaading the data for the further preprocessing and also for the preview 

@st.cache_data 
def load_data(): 
    return pd.read_csv("movies.csv") 

movies_data = load_data()  

# Dataset Preview 
st.subheader("📂 Dataset Preview") 
st.dataframe(movies_data.head(10)) 

# Now, we will be performing the data preprocessing  

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director', 'overview'] 

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('') 

# Combinig all the fatures by using the combined features aggregate function 
combined_features = movies_data[selected_features].agg(' '.join, axis = 1)  

# TFIDF + Cosine similarity 

tfidf = TfidfVectorizer(max_features = 3000, stop_words = 'english')  
feature_vectors = tfidf.fit_transform(combined_features) 

similarity = cosine_similarity(feature_vectors) 

movie_name = st.text_input("Enter your favourite movie name:")

# Recommend Function for thw whole code 

def recommend(movie_name):

    list_of_all_titles = movies_data['title'].tolist() 
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles) 

    if not find_close_match: 
        st.error("Movie not found!") 
        return None 
    
    close_match = find_close_match[0] 

    index_of_the_movie =  movies_data[movies_data.title == close_match].index[0] 

    similarity_score = list(enumerate(similarity[index_of_the_movie])) 
    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

    st.subheader("✨ Recommended Movies")  

    names = [] 
    scores = [] 

    i = 1 
    for movie in sorted_similar_movies: 
        index = movie[0] 
        title = movies_data.iloc[index]['title']  

        if i < 15:  
            st.write(f"⭐ {i}. {title}") 
            names.append(title)
            scores.append(movie[1]) 
            i +=1  # Here this line is compulsory for increment counter and limiting the movie names to 15 movies 
    return names, scores, index_of_the_movie  


if st.button("Recommend"):  

    result = recommend(movie_name) 

    if result: 
        names, scores, movie_index   =  result 

        # Bar chart 

        st.subheader("📊 Similarity Scores") 

        fig, ax = plt.subplots(figsize = (10,6)) 
        ax.barh(names[::-1], scores[::-1])  # Here we reverse the list as because the horizontal bar shows the reversed list, so reversing prints same
        ax.set_xlabel("Similarity") 
        st.pyplot(fig)  

        # heatmap  

        st.subheader("🔥 Cosine Similarity Heatmap") 

        sample =  similarity[movie_index][:20] 

        fig2, ax = plt.subplots(figsize = (12,4)) 
        sns.heatmap([sample], cmap = "YlOrRd", annot=False) 
        st.pyplot(fig2) 

st.markdown("---------------") 
st.caption("Built by Chaitanya Mangla using the Tfidf Vectorizer and Cosine Similarity")




