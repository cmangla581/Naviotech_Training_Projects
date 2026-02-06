
# Here in thsi swe are going to make the breast cancer prediction app on streamlit . 
# The Breast Cancer Predictor Dataset is imported from teh scikit learn library. 

import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer  
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Page configuration of Streamlit  

st.set_page_config(page_title ="Breast Cancer Predictor", layout = "centered")  

st.title("🩺 Breast Cancer Detection") 
st.write("Predict whether a tumor is Benign or Malignant using a Random Forest Classifier") 

# loading the dataset 
data = load_breast_cancer() 
x = pd.DataFrame(data.data,  columns = data.feature_names) 
y = pd.Series(data.target) 

# train, test and split the model of the machine learning algorithm 
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 42, test_size = 0.20)  

# training the Random Forest Classifier Model 
model = RandomForestClassifier(n_estimators = 500, random_state= 42) 
classifier = model.fit(x_train, y_train)  

# Model Accuracy 
y_pred = model.predict(x_test) 
accuracy = accuracy_score(y_test, y_pred) 

st.subheader("📊 Model Performance") 
st.success(f"✅ Accuracy:{accuracy*100:.2f}%") 

# user Input Section 
st.subheader("🔢 Enter the Tumor Features") 
st.write("Adjust the values below to make prediction:")

input_data = [] 

for feature in data.feature_names: 
    value = st.number_input(
        label = feature, 
        min_value = float(x[feature].min()),
        max_value = float(x[feature].max()),
        value = float(x[feature].mean()) 
    )
    input_data.append(value) 

# Convert the  input to numpy array 
input_array = np.array(input_data).reshape(1,-1) 

# prediction  

if st.button("🔍 Predict"):
    prediction = model.predict(input_array) 
    prediction_proba = model.predict_proba(input_array) 

    if prediction[0] == 1: 
        st.success("✅ Prediction: Benign (Non-cancerous)") 

    else: 
        st.error("⚠️ Prediction: **Malignant (Cancerous)")

    st.write(
        f"Benign: {prediction_proba[0][1] * 100:.2f}%, "
        f"Malignant:{prediction_proba[0][0] * 100:.2f}%"
    ) 

# Plotting of the confusion Matrix Plotted on the streamliit 
st.subheader("🧩 Confusion Matrix") 

cm = confusion_matrix(y_test, y_pred) 

fig, ax = plt.subplots() 

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Malignant", "Benign"]) 

disp.plot(ax=ax, cmap = "inferno", colorbar = False) 

ax.set_facecolor("black")
fig.patch.set_facecolor("black") 
ax.title.set_color("white") 
ax.xaxis.label.set_color("white") 
ax.yaxis.label.set_color("white") 

st.pyplot(fig)  

# Addition oof a footer 
st.markdown("--------")
st.caption("Built with Streamlit and Random Forest Classifier") 

