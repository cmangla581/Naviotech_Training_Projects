# Now, we will be making a mart sales predictior for the Sales Prediction Data Analysis Project for making it easy for the users 

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Big Mart Sales Predictor", layout="centered") 

# ---------------- STYLING ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #000000, #0f0f0f, #111111, #000000);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
}

@keyframes gradient {
    0% {background-position:0% 50%}
    50% {background-position:100% 50%}
    100% {background-position:0% 50%}
}

h1 {
    color: white !important;
    text-align: center;
    font-size: 55px;
    text-shadow: 0px 0px 20px cyan;
}

h2, h3, label, p, span {
    color: white !important;
}
            
div.stButton > button {
    background: linear-gradient(45deg, #00ffff, #00ffcc);
    color: black;
    font-size: 20px;
    font-weight: bold;
    padding: 12px 30px;
    border-radius: 12px;
    border: none;
    box-shadow: 0px 0px 15px cyan;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0px 0px 30px #00ffff;
    background: linear-gradient(45deg, #00ffcc, #00ffff);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df_original = pd.read_csv("Big_mart_sales.csv") 

df_original["Item_Fat_Content"] = df_original["Item_Fat_Content"].replace({
    "LF": "Low Fat",
    "low fat": "Low Fat",
    "reg": "Regular"
})

# Drop ID column
df_original.drop(["Item_Identifier", "Outlet_Identifier"], axis=1, inplace=True) 

# Fill missing values
df_original.fillna(df_original.mode().iloc[0], inplace=True)

# Copy for training
df = df_original.copy()

# ---------------- ENCODING ----------------
encoders = {}
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------- TRAIN MODEL ----------------
X = df.drop("Item_Outlet_Sales", axis=1)
y = df["Item_Outlet_Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ---------------- TITLE ----------------
st.markdown("<h1>🛒 BIG MART SALES PREDICTOR</h1>", unsafe_allow_html=True)

# ---------------- MENU ----------------
menu = st.selectbox(
    "Choose Feature",
    ["Prediction", "Correlation Heatmap", "Actual vs Predicted"]
)


if menu == "Prediction":

    st.header("Enter Product Details")

    user_input = {}

    for col in X.columns:

        # CATEGORICAL INPUTS (REAL VALUES)
        if col in cat_cols:
            options = sorted(df_original[col].unique())
            value = st.selectbox(col, options)

            # encode safely
            encoded_value = encoders[col].transform([str(value)])[0]
            user_input[col] = encoded_value

        # NUMERICAL INPUTS
        else:
            value = st.number_input(
                col,
                float(df_original[col].min()),
                float(df_original[col].max()),
                float(df_original[col].mean())
            )
            user_input[col] = value

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Sales"):
        result = model.predict(input_df)[0]
        st.markdown(f"""
<div style="
    background: rgba(0,255,255,0.1);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 25px cyan;
    border: 2px solid cyan;
    margin-top: 20px;
">
    <h2 style="color: white; text-shadow: 0px 0px 15px cyan;">
        💰 Predicted Sales
    </h2>
    <h1 style="
        font-size: 60px;
        color: #00ffff;
        text-shadow: 0px 0px 30px #00ffff;
    ">
         {round(result,2)}
    </h1>
</div>
""", unsafe_allow_html=True)
        

elif menu == "Correlation Heatmap":

    st.header("Feature Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


else:

    st.header("Actual vs Predicted Sales")

    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(y_test, predictions, color = 'red', label = 'Predictions') 
    ax.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 
         color = "green", linewidth = 2, label = "Prefect Line")  
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted")

    st.pyplot(fig) 