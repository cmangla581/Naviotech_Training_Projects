
# Now,we will be making the streamlit app for the Medical insurance Predictior, for the dataset analysis which we have done on
# Jupyter notebook  

import streamlit as st 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score   

# now, we will be setting the page configuration as; 

st.set_page_config(page_title = "Medical Insurance Predictor", layout = "centered")  

# Now  we will be applying the custom Css on the streamlit app 

st.markdown("""
<style>

/* ===== APP BACKGROUND ===== */
.stApp {
    background-color: black;
    color: white;
}

/* ===== TITLE GLOW ===== */
.glow {
    font-size: 55px;
    text-align: center;
    color: white;
    text-shadow:
        0 0 10px cyan,
        0 0 20px cyan,
        0 0 40px cyan,
        0 0 80px cyan;
}

/* ===== LABEL HEADINGS ===== */
label {
    color: white !important;
    font-weight: 600 !important;
}

/* ===== NUMBER INPUT ===== */
.stNumberInput input {
    background-color: #111 !important;
    color: white !important;
}

/* ===== SELECTBOX BACKGROUND ===== */
.stSelectbox div[data-baseweb="select"] {
    background-color: #111 !important;
}

/* ⭐⭐⭐ MOST IMPORTANT FIX ⭐⭐⭐ */
/* THIS MAKES SELECTED TEXT VISIBLE */
.stSelectbox div[data-baseweb="select"] input {
    color: white !important;
    -webkit-text-fill-color: white !important;
}

/* ===== DROPDOWN MENU ===== */
div[data-baseweb="popover"] {
    background-color: white !important;
}

/* ===== DROPDOWN OPTIONS ===== */
ul[role="listbox"] li {
    color: black !important;
    background-color: white !important;
}

/* ===== OPTION HOVER ===== */
ul[role="listbox"] li:hover {
    background-color: #e6f7ff !important;
    color: black !important;
}

/* ===== BUTTON STYLE ===== */
div.stButton > button {
    background-color: black;
    color: white;
    border: 2px solid cyan;
    padding: 12px;
    font-size: 18px;
    border-radius: 10px;
    box-shadow: 0 0 20px cyan;
}

div.stButton > button:hover {
    box-shadow: 0 0 50px cyan;
    transform: scale(1.08);
}

</style>
""", unsafe_allow_html=True)
# Now, we will be applying the custom css 

st.markdown("<h1 class = 'glow'> 💊 Medical Insurance Cost Predictor</h1>", unsafe_allow_html = True)  

# Now, we are loading the data 

data = pd.read_csv("medical_insurance.csv")  
st.subheader("📊 Dataset Preview") 
st.dataframe(data.head())  

# Now, we will perform the Label Encoder  
le = LabelEncoder()  
for col in ["sex", "smoker","region"]: 
    data[col] = le.fit_transform(data[col]) 

# now, we will be dividing into the features and targets 

x = data.drop(["charges"], axis = 1) 
y = data.filter(["charges"], axis = 1) 

# Now, we will be applying the train,test and split model 

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.2) 

# Now, we are applying the standard Scaling feature on the model 
sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test) 

# Now, we apply the model training on the dataset as 
model = RandomForestRegressor(n_estimators = 500, random_state = 42) 
regressor = model.fit(x_train, y_train) 

y_pred = regressor.predict(x_test) 

r2 = r2_score(y_test, y_pred)  

# Now, we willl be showing the model perfformance using a bit of css styling 
st.markdown("## 📈 Model Performance")

st.markdown(f"""
<h2 style='text-align:center; color:lime; text-shadow:0 0 20px lime;'>
R² Score: {round(r2,4)}
</h2>
""", unsafe_allow_html=True)  

# Now, here we have the input section, where all the input will be put for making the prediction 

st.markdown("## 🧾 Enter the Customer Details") 

col1, col2,  col3 = st.columns(3)  

with col1: 
    age = st.number_input("Age", 18,90,25) 
    bmi = st.number_input("BMI", 15.0,40.0,25.0) 

with col2: 
    children = st.number_input("Children", 0,5,1) 
    sex = st.selectbox("Sex", ["male", "female"]) 

with col3: 
    smoker = st.selectbox("Smoker", ["yes", "no"])  
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"]) 

# Encoding the inputs  
sex = 1 if sex == "male" else 0  
smoker = 1 if smoker == "yes" else 0 

region_map = {"southwest": 3, "southeast": 2, "northwest": 1, "northeast": 0} 
region = region_map[region]  

# Now we will input the data as; 
input_data = np.array([[age,sex, bmi, children, smoker, region]])
input_scaled = sc.transform(input_data)  

# Now, we will be making the predictions as: 

if st.button("✨ Predict Insurance Cost"): 
    prediction = regressor.predict(input_scaled)[0]  

    st.markdown(f"""
    <h2 style='text-align:center; color:cyan; text-shadow:0 0 20px cyan;'>
    Predicted Insurance Cost: $ {round(prediction,2)}
    </h2>
    """, unsafe_allow_html=True)  

# Nw, we will be plotting the visualization plots for the better understanding of the Ml Project  

st.markdown("<h1 class='glow'>📊 Data Visualizations</h1>", unsafe_allow_html=True) 

# Plotting the Actual vs Predicted Graph 

st.subheader("Actual vs Predicted Graph")

fig1 = plt.figure(figsize = (7,5), facecolor = "black")  

# Scatter Plots 
plt.scatter(y_test, y_pred) 

# Perfect Line 
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", label="Perfect Line")  

plt.xlabel("Actual Charges", color = "white") 
plt.ylabel("Prediction Charges", color = "white") 
plt.title("Actual vs Predicted", color = "white") 
plt.xticks(color = "white") 
plt.yticks(color = "white") 
plt.legend() 

st.pyplot(fig1)  

# Now, comes the plotting of the correlation heatmap 
st.subheader("Correlation Heatmap")

fig2 = plt.figure(figsize = (10,6), facecolor = "black") 
sns.heatmap(data.corr(), annot = True, cmap = "coolwarm") 
plt.title("Feature Correlation", color="white")
plt.xticks(color="white")
plt.yticks(color="white")
st.pyplot(fig2)

# Now, we will be plotting the charges distribution pplot  

st.subheader(" Charges Distribution") 

fig3 = plt.figure(facecolor = "black") 
sns.histplot(data["charges"], kde = True) 
plt.title("Insurance Charges Distribution", color = "white") 
plt.xticks(color = "white") 
plt.yticks(color = "white") 
st.pyplot(fig3) 

st.markdown("-----------")
st.caption("Built by Chaitanya Mangla using Streamlit and Random Forest Regressor") 