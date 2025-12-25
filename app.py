import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

## Load the pre-trained model and scalers
model=load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_gender.pkl', 'rb') as f:
    label_gender = pickle.load(f)

with open('onehotencoder.pkl', 'rb') as f:
    onehotencoder = pickle.load(f)



##streamlit app
st.title("Customer Churn prediction")
##user input
geography=st.selectbox("Geography",onehotencoder.categories_[0])
gender=st.selectbox("Gender",label_gender.classes_)
age=st.slider("Age",18,92)
balance=st.number_input("Balance")
credit_score=st.slider("Credit Score")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Number of Products",1,4)
has_cr_card=st.selectbox("Has Credit Card",[0,1])
is_active_member=st.selectbox("Is Active Member",[0,1])

##prepare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_gender.transform([gender])[0]],
    'Age':[age],    
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

##GEography
geo_encoded=onehotencoder.transform([[geography]])
geo_df=pd.DataFrame(geo_encoded,columns=onehotencoder.get_feature_names_out(['Geography']))

#combine the data
input_data=pd.concat([input_data.reset_index(drop=True),geo_df.reset_index(drop=True)],axis=1)

##scale the data
scaled_data=scaler.transform(input_data)

##prediction churn
prediction=model.predict(scaled_data)
prediction_proba=prediction[0][0]

if prediction_proba>0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_proba:.2f}")
else:
    st.write(f"The customer is unlikely to churn with a probability of {1-prediction_proba:.2f}")   

