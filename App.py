from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
from PIL import Image

model = load_model('df_diabetes')


 
def predict(model, input_df):
    predictions_df= predict_model(estimator=model, data=input_df)
    predictions = predictions_df.iloc[0]['prediction_label']
    return predictions

def run():
    image_hospital = Image.open('hospital.jpg')
    st.title("This is a diabetics dataset")
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online","Batch"))
    if add_selectbox=='Online':
        Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
        Glucose = st.number_input('Glucose', min_value=0, max_value=250, value=100)
        BloodPressure = st.number_input('BloodPressure', min_value=0, max_value=150, value=70)
        SkinThickness = st.number_input('SkinThickness', min_value=0, max_value=100, value=20)
        Insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
        bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
        
        DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=3.0, value=0.5)
        Age = st.number_input('Age', min_value=18, max_value=100, value=30)
        output=""
 
        input_dict = {'Pregnancies':Pregnancies ,'Glucose':Glucose,'BloodPressure':BloodPressure,'SkinThickness':SkinThickness,'Insulin': Insulin,'BMI':bmi,"DiabetesPedigreeFunction":DiabetesPedigreeFunction,'Age':Age}
        input_df=pd.DataFrame([input_dict])




        if st.button("Predict"):
            output=predict(model=model, input_df=input_df)
            output = " "+str(output)
        st.success('This Output is {}'.format(output))
 
    if add_selectbox=='Batch':
        file_upload = st.file_uploader("Upload a csv file for prediction", type=['csv'])

        if file_upload is not None:
            data= pd.read_csv(file_upload)
            predictions = predict_model(estimator = model, data=data)
            st.write(predictions)
 
 
 
if __name__ =='__main__':

    run()
