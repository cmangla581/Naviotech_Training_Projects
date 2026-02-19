'''
The Backend Analysis of the Pima Indian Diabetes Dataset is done on the Jupyter Notebook 

Now comes the part of Prediction using the Streamlit and the Random Forest Classifier Algorithm  

This is done to make a proper user Interface for the Dataset Analysis which can be easily used to make predictions 
''' 
# Importing the Libraries 
import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 

# Setting the Page Configuration 

st.set_page_config(page_title = "Diabetes Predictor", layout = "centered") 

st.title("🩺 Pima Indians Diabetes Dataset Prediction") 
st.write("Predict whether a person is diabetic using a Random Forest Classifier") 

# Loading  of the Dataset  

@st.cache_data
def load_data(): 
    data= pd.read_csv("diabetes.csv") 
    return data 
data =load_data() 

# Splitting of the Features and targets 
x = data.drop("Outcome", axis = 1) 
y = data["Outcome"]  

# Train, test and split appiied on the dataset 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.20, random_state = 0) 


# now comes the part of te feature scaling using teh Standard Scaler Library 
sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test) 

# Training of teh Random Forest Classifier ML Algorithm  
model = RandomForestClassifier(random_state = 42, n_estimators = 500) 
classifier = model.fit(x_train, y_train)  

# Side Inputs  
st.sidebar.header("Enter Patient Details") 

pregnancies = st.sidebar.number_input("Pregnancies", 0,20,1) # Here there can be values from 0 to 20 abd by default the value is 1 
glucose = st.sidebar.number_input("Glucose", 0,200, 120) 
blood_pressure = st.sidebar.number_input("Blood Pressure", 0,300,70)
skin_thickness = st.sidebar.number_input("Skin Thickness", 0,100,20) 
insulin = st.sidebar.number_input("Insulin", 0,900,80) 
bmi = st.sidebar.number_input ("BMI", 0.0,70.0,25.0) 
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5) 
age = st.sidebar.number_input("Age", 1,120,33) 

# Prediction from the data  

input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]) 

# Scaling the input  
input_scaled = sc.transform(input_data) 


if st.button("Predict"): 
    prediction = classifier.predict(input_scaled) 
    probability = classifier.predict_proba(input_scaled)[0][1] 

    if prediction[0] == 1 : 
        st.error(f"⚠️ Diabetic (Probability: {probability:})") 

    else: 
        st.success(f"✅ Not Diabetic (Probability: {probability:})") 



# Model Accuracy 
accuracy = model.score(x_test, y_test) 
st.write(f"Model Accuracy: {accuracy:.2f}")   
