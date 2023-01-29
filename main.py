import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json
st.header("Heart Attack Prediction App")
name = st.text_input("Enter your Name: ", key="name")
data = pd.read_csv(r"heart.csv")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='binary_crossentropy',
                     optimizer='adam', metrics=["accuracy"])
input_age = st.slider('Age', 0.0, float(max(data["age"])), 1.0)

st.write("1: Male \n0: Female")

input_sex = st.slider('sex', 0.0, float(max(data["sex"])), 1.0)
input_cp = st.slider('CP', 0.0, float(max(data["cp"])), 1.0)
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

prediction=0
if st.button('Make Prediction'):
    #inputs = np.expand_dims(
        #[input_age, input_sex, input_trtbps, input_chol, input_fbs, input_restecg, input_thalachh, input_exng, input_oldpeak, input_slp, input_caa, input_thall], 0)
    inputs = np.expand_dims([40,1,0,500,500,0,0,500,1,2,1,0,3],0)
    prediction = loaded_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write("Your chances are " + str(prediction*100) + "%")
