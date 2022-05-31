import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('clean_diabetes_data.csv')

st.title('Diabetes Status In Women Prediction App')
st.write("""
# Diabetes Prediction App

This app predicts the **Diabetes Status** in women!

Data obtained from the [palmerpenguins library](https://datahub.io/machine-learning/diabetes) by Vincent Sigillito.
""")
st.subheader('Training Data')
st.write(df.head())
st.write(df.describe())

st.subheader('Visualisation')
st._legacy_bar_chart(df)

x = df.drop(['results'], axis = 1)
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

st.sidebar.header('User Input Features')
def user_features():
        pregnancies = st.sidebar.slider('Pregnancies', 0, 17)
        glucose = st.sidebar.slider('Glucose level', 0,200)
        bloodpressure = st.sidebar.slider('Blood Pressure', 0, 122)
        skinthickness = st.sidebar.slider('Skin Thickness', 0, 100 )
        insulin = st.sidebar.slider('Insulin', 0, 846)
        bmi = st.sidebar.slider('BMI', 0, 67)
        DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4)
        age = st.sidebar.slider('Age', 21, 88)

        user_features = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': bloodpressure,
                'SkinThickness': skinthickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': age
}
        report_data = pd.DataFrame(user_features, index=[0])
        return report_data

user_data = user_features()

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

reg = LogisticRegression()
reg.fit(x_train, y_train)

st.subheader('Accuracy')
st.write(str(accuracy_score(y_test, reg.predict(x_test))*100)+'%')

user_result = reg.predict(user_data)
st.subheader('Your Report: ')
output = ''
if user_result[0]==0:
    output = 'You Likely to Test Negative For Diabetes'
else:
    output = 'You Likely to Test Positive For Diabetes'

st.write(output)


