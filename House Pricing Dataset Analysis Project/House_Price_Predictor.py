'''
California House Prce Predictor Streamlit App 

In the Jupyter notebook, we did the Notebook Analysis but here in this we will  make a streamlit app for making predictions 

This will be a useful sreamlit app using the Random Forest Regressor 
'''  

import streamlit as st 
import numpy as np 
import pandas as pd 


# importing the california housing dataset 
from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import r2_score 

# Page Configuration  

st.set_page_config(page_title = "California House Price Predictor", page_icon ="🏠" , layout =  "centered") 

st.title("🏠 California House Price Predictor") 
st.write("Predict Median House Value using the Random Forest Regressor") 

# Loading the dataset 
@ st.cache_data 
def load_data(): 
    data = fetch_california_housing (as_frame = True) 
    x= data.data 
    y = data.target 
    return x, y

x,y = load_data() 


# Now comes the training the model  
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)  

# Now comes the scaling of the  ML Algorithm  
sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test)  

# Now, we train the Random Forest Regressor for making the predictions  
model = RandomForestRegressor(n_estimators = 500,random_state = 42, n_jobs = -1) 

classifier = model.fit(x_train, y_train)  

# Evaluation  
y_pred = classifier.predict(x_test) 
r2= r2_score(y_test, y_pred)


st.success(f"✅ Model R² Score: {r2:.2f}")  


# Now we put the sidebar inputs 
st.sidebar.header("🔧 Enter House Details") 

MedInc = st.sidebar.slider("Median Income (MedInc)", 0.5,15.0, 3.0) 
HouseAge = st.sidebar.slider("House Age", 1.0,60.0,20.0) 
AveRooms = st.sidebar.slider("Average Rooms",1.0,10.0,5.0) 
AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0,1.0) 
Population =  st.sidebar.slider("Population", 100.0,5000.0,1500.0) 
AveOccup = st.sidebar.slider("Average Occupancy", 1.0, 6.0, 3.0) 
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0,34.0) 
Longitude = st.sidebar.slider("Longitude", -124.0,-114.0,-118.0) 

# Now, we make the predictions using the machine lerning 
input_data = np.array([[MedInc,HouseAge, AveRooms,AveBedrms, Population,AveOccup, Latitude, Longitude]]) 

input_scaled = sc.transform(input_data) 

if st.button("🔮 Predict House Price"): 
    prediction = classifier.predict(input_scaled)[0]  

    st.subheader("📊 Prediction Result") 
    st.metric(
        label = "Estimated Median House Value",
        value=f"${prediction* 100000:,.0f}" 
    ) 



st.caption("Built by Chaitanya Mangla using the Streamlit and the Random Forest Regressor") 



