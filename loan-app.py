  
import streamlit as st
import pandas as pd
import numpy as np
import logging
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.ensemble import RandomForestClassifier


image = Image.open('banner.jpg')
st.image(image, use_column_width = True, output_format="JPG")


st.write("""
Il s'agit d'un projet dont le but est de définir si l'on doit accorder un prêt ou non en fonction de spécifications d'entrées.
""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe
#uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
uploaded_file = None

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Gender = st.sidebar.selectbox('Gender',('Male','Female'))
        Married = st.sidebar.selectbox('Married',('No','Yes'))
        Dependents = st.sidebar.selectbox('Dependents',('0', '1', '2', '3+'))
        Education = st.sidebar.selectbox('Education',('Graduate', 'Not Graduate'))
        Self_Employed = st.sidebar.selectbox('Self_Employed',('No', 'Yes'))
        ApplicantIncome = st.sidebar.number_input('ApplicantIncome', 0, 100000, 0)
        CoapplicantIncome = st.sidebar.number_input('CoapplicantIncome', 0, 50000, 0)
        LoanAmount = st.sidebar.number_input('LoanAmount', 0, 1000, 0)
        Loan_Amount_Term = st.sidebar.slider('Loan_Amount_Term (Year)', 1, 40)
        Credit_History = st.sidebar.selectbox('Credit_History', (0.0, 1.0))
        Property_Area = st.sidebar.selectbox('Property_Area', ('Urban', 'Rural', 'Semiurban'))
     

        data = {'Gender': Gender,
                'Married': Married,
                'Dependents': Dependents,
                'Education': Education,
                'Self_Employed': Self_Employed,
                'ApplicantIncome': ApplicantIncome,
                'CoapplicantIncome': CoapplicantIncome,
                'LoanAmount': LoanAmount,
                'Loan_Amount_Term': Loan_Amount_Term * 12,
                'Credit_History': Credit_History,
                'Property_Area': Property_Area,
                #'SumperMount' : LoanAmount / Loan_Amount_Term
                }

        logging.warning(data)

        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
df_raw = pd.read_csv('train.csv')
selection = df_raw.drop(columns=['Loan_Status'])
df = pd.concat([input_df,selection],axis=0)
df = df.drop(columns=['Loan_ID'])
df1 = df.copy()



logging.warning('OK pour df1 !')
# traitement du dataframe
df1["Dependents"] = df1["Dependents"].astype(str)


df1["Gender"].fillna(df["Gender"].mode()[0],inplace=True)
df1["Married"].fillna(df["Married"].mode()[0],inplace=True)
df1["Self_Employed"].fillna(df["Self_Employed"].mode()[0],inplace=True)
df1["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0],inplace=True)
df1["Dependents"].fillna(df["Dependents"].mode()[0],inplace=True)
df1["Credit_History"].fillna(df["Credit_History"].mode()[0],inplace=True)
df1["LoanAmount"].fillna(df["LoanAmount"].median(),inplace=True)


# encodage des labels
le = LabelEncoder()
df1["Dependents"] = le.fit_transform(df1["Dependents"])
logging.warning('1')
df1["Gender"] = le.fit_transform(df1["Gender"])
logging.warning('2')
df1["Married"] = le.fit_transform(df1["Married"])
logging.warning('3')
df1["Self_Employed"] = le.fit_transform(df1["Self_Employed"])
logging.warning('4')
df1["Education"] = le.fit_transform(df1["Education"])
df1["Property_Area"] = le.fit_transform(df1["Property_Area"])
#df1["Loan_Status"] = le.fit_transform(df1["Loan_Status"])
logging.warning('5')


logging.warning(df1.columns)



#Utilisation de la colonne log() pour tasser les valeurs et diminuer l'impact des outliers
df1["ApplicantIncome"] = np.log(df1["ApplicantIncome"])
#Comme la colonne "CoapplicantIncome" a quelques valeurs "0", nous obtiendrons des valeurs logarithmiques sauf "0".
df1["CoapplicantIncome"] = [np.log(i) if i!=0 else 0 for i in df1["CoapplicantIncome"]]
df1["LoanAmount"] = np.log(df1["LoanAmount"])


df = df1[:1] # Selects only the first row (the user input data)
logging.warning(df)


# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('loan_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
loan_prediction = np.array(['loan refused', 'loan accorded'])
st.write(loan_prediction[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)