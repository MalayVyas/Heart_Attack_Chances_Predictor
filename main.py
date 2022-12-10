import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
st.header("Heart Attack Prediction App")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("https://github.com/MalayVyas/Heart_Attack/blob/main/heart.csv", sep=";")
# load label encoder
# encoder = LabelEncoder()
# encoder.classes_ = np.load('classes.npy', allow_pickle=True)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    data


input_age = st.slider('Age', 0.0, float(max(data["age"])), 1.0)

st.write("1: Male \n0: Female")

input_sex = st.slider('sex', 0.0, float(max(data["sex"])), 1.0)
input_trtbps = st.slider('trtbps', 0.0, float(max(data["trtbps"])), 1.0)
input_chol = st.slider('Cholestrol', 0.0, float(max(data["chol"])), 1.0)
input_fbs = st.slider('fbs', 0.0, float(max(data["fbs"])), 1.0)
input_restecg = st.slider('restecg', 0.0, float(max(data["restecg"])), 1.0)
input_thalachh = st.slider('thalachh', 0.0, float(max(data["thalachh"])), 1.0)
input_exng = st.slider('exng', 0.0, float(max(data["exng"])), 1.0)
input_oldpeak = st.slider('oldpeak', 0.0, float(max(data["oldpeak"])), 1.0)
input_slp = st.slider('slp', 0.0, float(max(data["slp"])), 1.0)
input_caa = st.slider('caa', 0.0, float(max(data["caa"])), 1.0)
input_thall = st.slider('thall', 0.0, float(max(data["thall"])), 1.0)


if st.button('Make Prediction'):
    # input_species = encoder.transform(np.expand_dims(inp_species, -1))
    inputs = np.expand_dims(
        [input_age, input_sex, input_trtbps, input_chol, input_fbs, input_restecg, input_thalachh, input_exng, input_oldpeak, input_slp, input_caa, input_thall], 0)
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your fish Chances are: {np.squeeze(prediction, -1):.2f}g")

