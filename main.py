import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json
st.header("Heart Attack Prediction App")
data = pd.read_csv("heart.csv")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='binary_crossentropy',
                     optimizer='adam', metrics=["accuracy"])
input_age = st.slider('Age', 0, 100, 1)

st.write("1: Male \n0: Female")
input_sex = 0
if st.checkbox(label="Male"):
    input_sex = 1
if st.checkbox(label="Female"):
    input_sex = 0

input_cp = st.slider('CP', 0.0, float(max(data["cp"])), 1.0)
input_trtbps = st.slider('trtbps', 0, (max(data["trtbps"])), 1)
input_chol = st.number_input(label="Cholestrol", min_value=0, max_value=400)
st.write("Fasting Blood Sugar")
input_fbs = 0
if st.checkbox(label="Yes, There is a Fasting Sugar problem."):
    input_fbs = 1

st.write("RestECG")
input_restecg = 0
if st.checkbox(label="Yes, There is an abnormailty in the ECG"):
    input_restecg = 1


input_thalachh = st.slider('thalachh', 0, (max(data["thalachh"])), 1)
st.write("EXANG")
input_exng = 0
if st.checkbox(label="Yes, There is Exercise Induced Angina"):
    input_exng = 1
    
input_oldpeak = st.slider('oldpeak', 0.0, float(max(data["oldpeak"])), 1.0)
input_slp = st.slider('slp', 0, (max(data["slp"])), 1)
input_caa = st.number_input(label="CAA", min_value=0, max_value=5)
st.write("Person's maximum Heart rate Level acheived:")
input_thalachh = st.number_input(label="Heart Rate Level",min_value=0, max_value=3)


if st.button('Make Prediction'):
    inputs = np.expand_dims(
        [input_age, input_sex, input_cp, input_trtbps, input_chol, input_fbs, input_restecg, input_thalachh, input_exng, input_oldpeak, input_slp, input_caa, input_thalachh], 0).astype(np.float)
    prediction = loaded_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your Heart Attack chancesare: {prediction[0][0]}")
