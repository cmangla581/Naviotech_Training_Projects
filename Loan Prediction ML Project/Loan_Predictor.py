
# here, I will be deploying the Loan Predictor App on Streamlit to give it a perfect look and easily usable by the users 

import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix 

# Setting the page Configuration 
st.set_page_config(page_title = "Loan prediction App", layout = "centered") 

# now, we will be making the custom css to give 

st.markdown("""
<style>

/* ------------- Main Background----------- */ 
.main {
     background-color: #fff9db;                       
} 

/*------------- Full Page Background-----------*/ 
.stApp{
    background: linear-gradient(135deg, #fff9db, #fff3b0);                             
    colr : #2b2b2b;
} 

/*------------Headings--------------*/ 
h1, h2, h3, h4 {
            color: #5a3e00; 
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.7);
}        
  
/*------------ Text Visibility------------*/ 
p,label,span{
            color: #2b2b2b !important;
            font-size: 16px;
}                     
            /* ----------- SIDEBAR ----------- */
[data-testid="stSidebar"] {
    background-color: #fff4c2;
}

/* ----------- INPUT BOXES ----------- */
.stNumberInput input {
    border-radius: 8px;
    border: 2px solid #f0c000;
}

/* ----------- SHINING BUTTON ----------- */
.stButton > button {
    background: linear-gradient(45deg, #ffd700, #ffb700);
    color: #3b2f00;
    font-weight: bold;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    border: none;
    box-shadow: 0px 4px 15px rgba(255, 215, 0, 0.6);
    transition: all 0.3s ease;
}

/* ----------- BUTTON HOVER EFFECT ----------- */
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0px 6px 20px rgba(255, 200, 0, 0.9);
}

/* ----------- PREDICTION RESULT BOX ----------- */
.pred-box {
    background: linear-gradient(135deg, #fff3b0, #ffe066);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: #3b2f00;
    box-shadow: 0px 4px 20px rgba(255, 215, 0, 0.7);
}

</style>
""", unsafe_allow_html=True) 

st.title("🏦 Loan Prediction ML Predictor") 

# loading the data in the model 

data = pd.read_csv("loan.csv") 

# Now, we will be dropping the Loan id and  the Gender columns, so as to carry out the further processing of the data 

data2  =  data.drop(["Loan_ID","Gender"], axis = 1)  

# Handle the missing values from the data 

for col in data2.columns: 
    if data2[col].dtype == "object": 
        data2[col].fillna(data2[col].mode()[0], inplace = True) 
    else: 
        data2[col].fillna(data2[col].median(), inplace = True) 

# Now, we will caarry out the encoding oof the whole data to convert the categorical into columns and also saving the maps 

label_maps = {} 

le = LabelEncoder() 
for col in data2.select_dtypes(include = "object").columns:
    le = LabelEncoder() 
    data2[col] = le.fit_transform(data2[col])
    label_maps[col] = dict(zip(le.classes_, le.transform(le.classes_))) 

# Splitting the data into the features and labels 

x = data2.drop(["Loan_Status"], axis = 1) 
y =  data2.filter(["Loan_Status"], axis = 1)   

# Noow,, we will be dividing the data into the train and test splits as: 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42) 

# standard ascaling oof the feature will be done as  
sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test)  

# Now, we will be training the model using thr KNeighborsClassifier  

model = KNeighborsClassifier(n_neighbors = 5) 
classifier = model.fit(x_train, y_train)

y_pred = classifier.predict(x_test) 
accuracy = accuracy_score(y_test, y_pred) 

st.success(f"Model Accuracy: {round(accuracy*100,2)}%") 

# Now, we will be putting the sidebar inputs, which will help in further predictions 

st.sidebar.header("Enter Applicant Details")  

user_input = [] 

for col in x.columns: 

    # categorical input 
    if col  in label_maps:  
        options =  list(label_maps[col].keys()) 
        selected = st.sidebar.selectbox(col, options) 
        encoded_val =  label_maps[col][selected] 
        user_input.append(encoded_val)  

    else: 
        val =  st.sidebar.number_input(col, value=float(x[col].mean())) 
        user_input.append(val)  

# Now, we will be carrying out the predictions as:  

if st.sidebar.button("Predict Loan Status"): 

    input_array = np.array(user_input).reshape(1,-1)  

    input_array = sc.transform(input_array) 

    prediction  =  classifier.predict(input_array) 
    probability =  classifier.predict_proba(input_array) 

    approval_prob =  probability[0][1] * 100 
    rejection_prob = probability[0][0] * 100 

    if prediction ==  1: 
        result = f"🎉 Loan Approved<br>Probability: {approval_prob:.2f}%" 

    else: 
        result = f"❌ Loan Rejected<br>Probability: {rejection_prob:.2f}%" 

    st.markdown(f"<div class='pred-box'>{result}</div>", unsafe_allow_html=True) 

# Now, the plotting of the correlation heat map, confusion matrix and the histogram will take place  

st.subheader("📊 Correlation Heatmap") 

fig1, ax1  =  plt.subplots(figsize  =(10,6)) 
sns.heatmap(data2.corr(), annot =  True, cmap = "RdBu", ax=ax1) 
st.pyplot(fig1) 

# Plotting of the Histogram for  visualization 

st.subheader("📈 Loan Status Histogram") 
fig3, ax = plt.subplots(figsize=(8,5)) 

sns.histplot(data2["Loan_Status"], kde = True, ax=ax) 
plt.title("Distribution of Loan Status") 

col_mean = data2["Loan_Status"].mean()
col_median = data2["Loan_Status"].median() 

ax.axvline(col_mean, color = "red", linestyle = "--", label = "Mean") 
ax.axvline(col_median, color = "green", linestyle = "-", label = "Median") 

ax.set_title("Loan Status Histogram") 
ax.legend()

st.pyplot(fig3)   

# Plotting the Property Area Grapj 

st.subheader("📈 Property Area Histogram") 
fig4, ax =  plt.subplots(figsize = (8,5)) 

sns.histplot(data2["Property_Area"], kde =  True, ax=ax)  
plt.title("Distribution of Property Area") 

col_mean = data2["Property_Area"].mean() 
col_median =  data2["Property_Area"].median() 

ax.axvline(col_mean, color = "red", linestyle = "--", label = "Mean") 
ax.axvline(col_median, color = "green", linestyle = "-", label = "Median") 

ax.set_title("Property Histogram") 
ax.legend() 

st.pyplot(fig4) 

st.markdown("------") 
st.caption("Built by Chaitanya Mangla using Streamlit and K Neighbors Classifier") 





