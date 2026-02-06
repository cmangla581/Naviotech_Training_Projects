'''
In the dataset analyisis, we had seen the analysis of the datset on the Jupyter Notebook. 

Now, here in this project i jave made a Streamlit Predictor for UI which can be easily used by the users. 
''' 

import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns  

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import  RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score 

# Setting the page configuration 
st.set_page_config(page_title = "Titanic Survival Predictor", page_icon = "🚢", layout = "centered")  

st.title("🚢 Titanic Survival Predictor") 
st.write("Random Forest Classification with Coorelation Heatmap") 

# Loading the dataset for the dataset analysis  

@st.cache_data 
def load_data(): 
    return pd.read_csv("Titanic_Dataset.csv")   

df = load_data() 

st.subheader("Dataset Preview") 
st.dataframe(df.head()) 

# Preprocessing the data 
data = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].copy() 

data["Age"].fillna(data["Age"].median(), inplace = True) 
data["Sex"] = data["Sex"].map({"male" : 0, "female" : 1}) # Here the manual  lebelling hase benn done for male and female 

x= data.drop("Survived", axis = 1) 
y = data["Survived"]


# Plotting of the Heatmap 
# Here we have to plot the feature coorelaation heatmap to show the coorelation 

st.subheader("Feature Correlation Heatmap") 

fig, ax = plt.subplots(figsize = (6,4)) 
sns.heatmap(data.corr(), annot = True, cmap = "coolwarm", ax=ax) 
st.pyplot(fig) 
plt.close(fig) 

# Now, we have to train, test and split the data 
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.2) 

# now, comes the scaling oof the figure 
sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test) 

# Model Training  

model = RandomForestClassifier(n_estimators = 500, random_state = 42)  
classifier = model.fit(x_train, y_train) 

# Now, we pperform the model evaluation as  

y_pred = classifier.predict(x_test) 
accuracy = accuracy_score(y_test, y_pred) 

st.subheader("Model Performance") 
st.success(f"✅ Accuracy: {accuracy:.2f}")   

# user Input Section where we can do the input of the details  

st.subheader("🔮 Predict Survival") 

Pclass = st.selectbox("Passenger Class", [1,2,3]) 
Sex = st.selectbox("Sex", ["male", "female"]) 
Age = st.slider("Age", 1,80,25)  
SibSp = st.number_input("Siblings / Spouses aboard", 0,8,0) 
Parch = st.number_input("Parents / Children aboard", 0,6,0) 
Fare = st.number_input("Fare", 0.0, 600.0, 50.0) 

Sex = 0 if Sex == "male" else 1   

user_input = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare]]) 
user_input_scaled = sc.transform(user_input) 

if st.button("Predict"): 
    prediction = classifier.predict(user_input_scaled)[0] 
    probability = classifier.predict_proba(user_input_scaled)[0][1] 

    if prediction == 1 : 
        st.success(f"🎉 Survived (Probability: {probability:.2f})") 

    else: 
        st.error(f"❌ Did Not Survive (Probability: {probability:.2f})") 

st.markdown("-----") 
st.caption("Built with Streamlit and Scikit-Learn") 



