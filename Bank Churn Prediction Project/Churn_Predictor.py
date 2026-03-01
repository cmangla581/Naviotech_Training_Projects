# Here, we will be amking a Streamlit Predictor for the Bank Churn Prediction Dataset Analysis project. 


import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns  

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 

# Now, we will be setting the page configuration as  
st.set_page_config(page_title = "Bank Churn Predictor", layout = "centered") 

# Now, we will also be adding the custom css to make the streamli app more attractive and 
st.markdown("""
<style>

/* ===== MAIN BACKGROUND ===== */
body, .stApp {
    background-color: #000000;
    color: white;
}

/* ===== GENERAL TEXT ===== */
html, body, [class*="css"]  {
    color: white !important;
}

/* ===== HEADINGS WITH GLOW EFFECT ===== */
h1, h2, h3, h4, h5, h6 {
    color: white !important;
    text-shadow: 
        0 0 5px #ffffff,
        0 0 10px #ffffff,
        0 0 20px #00ffff,
        0 0 30px #00ffff;
}

/* ===== SELECTBOX LABELS (FIX FOR DROPDOWN HEADINGS) ===== */
label {
    color: white !important;
    font-weight: bold;
}

/* Dropdown text */
div[data-baseweb="select"] {
    color: white !important;
}

/* Dropdown selected value */
div[data-baseweb="select"] span {
    color: white !important;
}

/* ===== BUTTON STYLING ===== */
.stButton>button {
    background-color: #111111;
    color: white;
    border-radius: 10px;
    border: 2px solid #00ffff;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #00ffff;
    color: black;
}

/* ===== SUCCESS MESSAGE ===== */
.stSuccess {
    background-color: #111111 !important;
    color: #00ffcc !important;
    border: 1px solid #00ffcc;
}

/* ===== ERROR MESSAGE ===== */
.stError {
    background-color: #111111 !important;
    color: #ff4d4d !important;
    border: 1px solid #ff4d4d;
}

/* ===== INPUT BOXES ===== */
input, textarea {
    background-color: #111111 !important;
    color: white !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background-color: #0a0a0a;
    color: white;
}

</style>
""", unsafe_allow_html=True)



# Now comes the main part of loading the model and training the data 

@st.cache_data 
def load_train():  

    df = pd.read_csv("Customer_Churn.csv") 

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = "coerce") 
    df.dropna(inplace = True) # Removing of all the missing values in the data 

    # Now, we will be taking the features for the data analysis  

    features = [
        "TotalCharges", "MonthlyCharges", "tenure", 
        "Contract", "InternetService", "PaymentMethod",
        "TechSupport", "OnlineSecurity", "PaperlessBilling" 

    ]

    x= df[features] 
    y = df["Churn"].map({"Yes":1, "No" : 0}) 

    x= pd.get_dummies(x) 

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42) 

    model = RandomForestClassifier(n_estimators= 500, random_state = 42) 
    model.fit(x_train, y_train)  

    # Printing the accuracy of the model as:  
    y_pred = model.predict(x_test)  
    acc = accuracy_score(y_test, y_pred) 

    return df, model, x.columns,acc 

df, model, model_columns, accuracy = load_train() 

# Now, we will be putting the title of the whole heading as: 

st.title("🏦 Bank Customer Churn Prediction") 

# Displaying the model acccuracy 
st.subheader("📈 Model Performance") 

st.success(f"✅ Model Accuracy:{accuracy*100:.2f}%")

st.markdown("----") 

# Now, we will be plotting he correlation heatmap of all the fatures, so as to show the relation between the fatures as: 

st.subheader("📊 Correlation Heatmap (All Features)") 

cols = [
    "TotalCharges", "MonthlyCharges", "tenure",
    "Contract", "InternetService", "PaymentMethod",
    "TechSupport", "OnlineSecurity", "PaperlessBilling",
    "Churn"
]  

data = df[cols].copy() 

# Encoding the categorical outcomes  
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

corr = data.corr()

fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

st.pyplot(fig) 


st.markdown("-----") 

# Prediction Section 

st.subheader("🔮 Predict Customer Churn") 
col1, col2 = st.columns(2) 

with col1: 
    TotalCharges = st.number_input("Total Charges", min_value = 0.0) 
    MonthlyCharges = st.number_input("Monthly Charges", min_value = 0.0) 
    tenure = st.number_input("Tenure (Months)", min_value = 0) 

    Contract =  st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"]) 

    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]) 

with col2: 
    PaymentMethod  =  st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit Card (automatic)"])

    TechSupport = st.selectbox("Tech Suppport", ["Yes", "No"]) 

    OnlineSecurity = st.selectbox("Online Security", ["Yes","No"])

    PaperlessBilling = st.selectbox("Paperless billing", ["Yes", "No"])  

# Prediction Button for the streamlit prediction can be done as: 

if st.button("🚀 Predict Churn Risk"): 

    input_df = pd.DataFrame({
        "TotalCharges": [TotalCharges],
        "MonthlyCharges" : [MonthlyCharges], 
        "tenure" : [tenure], 
        "Contract" : [Contract], 
        "InternetService" : [InternetService], 
        "PaymentMethod"  : [PaymentMethod], 
        "TechSupport" : [TechSupport], 
        "OnlneSecurity" : [OnlineSecurity],  
        "PaperlessBilling" : [PaperlessBilling]
    }) 

    input_df  = pd.get_dummies(input_df) 
    input_df = input_df.reindex(columns = model_columns, fill_value =  0) 

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] 

    st.markdown("----") 

    if prediction == 1: 
        st.error(f"⚠️ Customer is likely to CHURN\n\nRisk Score: {probability*100:.2f}%") 

    else: 
        st.success(f"✅ Customer will STAY\n\nConfidence: {(1-probability)*100:.2f}%")  

# Addition of a footer as: 
st.markdown("--------") 
st.caption("Built by Chaitanya Mangla using the Streamlit and Random Forest Classifier") 
