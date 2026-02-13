
# Here this is the whole Strealit Predictor using the Dataset of the Heart patients using the Logistic regression 
import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, accuracy_score 

# Setting the Page Configuration 
st.set_page_config(page_title = "Heart Disease Predictor", layout = 'centered')  


#  Anatomical heart icon and title 
st.markdown(""" 
<div style="text-align:center">
    <img src="https://cdn-icons-png.flaticon.com/512/2966/2966481.png" width="140">
    <h1 style="color:#ff4b4b;">Heart Disease Prediction System</h1>
</div>
""",  unsafe_allow_html=True) 

# Dark Theme CSS : 

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

/* Fix labels */
label {
    color: white !important;
    font-weight: 600;
}

/* Fix number input boxes */
div[data-baseweb="input"] input {
    color: white !important;
    background-color: #262730 !important;
}

/* Fix dropdowns */
div[data-baseweb="select"] {
    color: white !important;
}
            
div.stButton > button {
    background: linear-gradient(90deg, #ff0000, #ff4b4b) !important;
    color: white !important;
    font-size: 26px !important;
    font-weight: 900 !important;
    border-radius: 18px !important;
    padding: 16px 50px !important;
    letter-spacing: 1.5px;
    border: none;
    text-shadow: 0px 0px 10px black;
    box-shadow: 0px 0px 20px #ff4b4b;
    transition: 0.3s ease;
}

/* BUTTON HOVER EFFECT */
div.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0px 0px 40px #ff0000;
}
</style>
""", unsafe_allow_html=True)



# Loading the Data 

data = pd.read_csv("dataset_heart.csv") 
st.subheader("📊 Dataset Preview") 
st.dataframe(data.head())  

data.rename(columns={
    "chest pain type": "cp", 
    "resting blood pressure" : "rbp",
    "serum cholestoral" : "chl", 
    "fasting blood sugar" : "fbs", 
    "resting electrocardiographic results" : "rcg", 
    "max heart rate" : "hr", 
    "exercise induced angina" : "exanga", 
    "ST segment" : "slope", 
    "major vessels" : "vessels" 
}, inplace=True)

# Plotting of coorelation  
st.subheader("📊 Correlation Heatmap") 

fig,ax = plt.subplots(figsize = (14,10)) 
sns.heatmap(data.corr(), annot = True, cmap = "RdBu", ax =ax) 
st.pyplot(fig) 

# Now comes the dividing of the data into the features andlabels 
x = data.drop("heart disease", axis = 1) 
y= data["heart disease"]  

# Dividing into the train and test split 
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size=0.2) 

# Standard Scaling the features for further applying of ML algorithm 
sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test) 

# Applying the ML Algorithm  
model = LogisticRegression()
classifier = model.fit(x_train, y_train)  
y_pred = classifier.predict(x_test)  

# Uploading thr confusion Matrix 
st.subheader("📊 Confusion Matrix") 
cm = confusion_matrix(y_test, y_pred) 

fig2, ax2 = plt.subplots() 
sns.heatmap(cm, annot = True, fmt = 'd', cmap = "Blues", ax = ax2) 
ax2.set_xlabel("Predicted") 
ax2.set_ylabel("Actual") 
st.pyplot(fig2) 

acc = accuracy_score(y_test, y_pred) * 100 
st.success(f"✅ Model Accuracy: {acc:.2f}%")  

# Adding of the Features Description in the Section 

st.subheader("🩺 About the Feature Columns") 

st.markdown(""" 
This Heart Disease Prediction Model uses the following medical features: 
1. Age --> Age of the Patient in years 

2. Sex --> Gender (0: Female, 1: Male) 

3. Chest pain Type: 
   
   a. Typical Angina (High Risk)
   
   b. Atypical Angina (Moderate Risk)
   
   c. Non Anginal pain (Low Risk)
   
   d. Asymtomatic (Very Low or No Risk) 
            
4. Resting Blood Pressure : Blood Pressure measured at rest 
            
5. Serum Choolestoral : Amount of cholestoral present in the blood. 
            
6. Fasting blood Sugar : Blood Sugar level present after not eating for 8 hours. 

    1 >> Blood sugar above 120, which is high  

    0 >> Blood sugar below 120 which is moderate to low 

7. Resting Electrocardiographic Results : An ECG is basically the record of the electrical activity of the heart. 

    0 --> Normal ECG (Low) 
    
    1 --> ST Wave Abnormality (Moderate Risk)

    2 --> Left Ventricular Hypertrophy ---> High Risk 

8. Max Heart Rate : This is the maximum heart rate achieved.  

9. Exercise Induced Angina : It basically means the chest pain caused by the reduced blood flow to the heart. 

    1 >> High Risk 

    0 >> Low Risk 

10. Old Peak : It means that how much the heart's electrical signal changes when the heart is stressed.
    
    0-1 >> Normal : Low Risk 
    
    1-2 >> Mild Depression : Moderate Risk 
            
    More than 2 >> Significant Depression : High Risk  
            

11. ST Segment : Slope of peak exercise ST Segment. 
            
    1 >> Up Sloping ST Segment >> Low Risk 
            
    2 >> Flat ST Segment >> Moderate Risk  
            
    3 >> Downsloping ST Segment >> High Risk 
            

12. Major Vessels : Major Vessels blocked or visible in Fluroscopy 
    
    0 >> No Blood Vessls Blocked >> No Risk 
            
    1 >> One Blood Vessel Blokced >>  Low Risk 
            
    2 >> Two Blood Vessels Blocked >> Moderate Risk 
            
    3 >> Three Blood Vessels Blocked >> High Risk 
            

13. Thal : It refers to Thallium  Stress test that checks the blood flow to heart muscles.  
            
    3 >> Normal blood flow >> low risk 

    6 >> Fixed Detect >> Moderate Risk

    7 >>> Reverible Defect  >> High risk 

""" )  

# User Inpput Sections  

st.subheader("📋 Enter Patient Details")  

col1, col2, col3 = st.columns(3) 

with col1: 
    age = st.number_input("Age", 20, 100, 50) 
    sex = st.selectbox("Gender (0 = Female, 1 = Male)", [0,1]) 
    cp = st.selectbox("Chest Pain Type", [1,2,3,4]) 
    rbp = st.number_input("Resting BP", 80,200,120)  

with col2: 
    chl = st.number_input("Cholestoral", 100,600,200) 
    fbs = st.selectbox("Fasting blood Sugar", [0,1]) 
    rcg = st.selectbox("Rest ECG", [0,1,2]) 
    hr = st.number_input("Max Heart Rate", 60, 300, 150) 

with col3: 
    exanga = st.selectbox("Exercise induced Angina", [0,1]) 
    oldpeak = st.number_input("Old Peak", 0.0, 6.0, 1.0) 
    slope = st.selectbox("ST Segment", [1,2,3]) 
    vessels = st.selectbox("Major Vessels Blocked", [0,1,2,3]) 
    thal = st.selectbox("Thal", [3,6,7]) 


# Prediction Button 
if st.button("💊 Predict Heart Disease"): 

    patient = np.array([[age, sex,cp,rbp,chl,fbs,rcg,hr,exanga,oldpeak,slope,vessels,thal]]) 

    patient = sc.transform(patient) 

    prediction = classifier.predict(patient)[0] 

    probability = classifier.predict_proba(patient)[0][1] 

    if prediction == 2: 
        st.error(f"⚠ High Risk of Heart Disease\nProbability: {round(probability*100,2)}%") 

    else: 
        st.success(f"✅ Low Risk of Heart Disease\nProbability: {round(probability*100,2)}%") 
 

# Addition  of a footer 

st.markdown("--------")
st.caption("Built by Chaitanya Mangla using Streamlit and Logistic Regression") 






