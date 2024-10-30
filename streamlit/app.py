import streamlit as st
import pandas as pd

from src.config.config_variables import PredictionPipelineConfig
from src.pipelines.prediction_pipeline import PredictPipeline

PredictionPipelineConfig = PredictionPipelineConfig()

def predict (df):
        call_pipeline = PredictPipeline(PredictionPipelineConfig)
        predection_result = call_pipeline.predict(df)
        predection_result = predection_result.tolist()
        return round(predection_result[0],0)

st.title("Medical Insurance Cost Predictor")

with st.form(key="input_form"):
        
    col1, col2 = st.columns(2)
    age = col1.number_input("Age", min_value=18, max_value=64, value=40)
    sex = col1.selectbox(
        "Sex",
        ("male", "female"),
    )
    bmi = col1.number_input("BMI", min_value=16.0,max_value=53.0, value=30.6)
    children = col2.number_input("Children", min_value=0,max_value=4, value=2)
    smoker = col2.selectbox(
        "Smoker",
        ("yes", "no"),
    )
    region = col2.selectbox(
        "Region",
        ("northeast", "southeast","northwest", "southwest"),
    )
    if st.form_submit_button("Predict"):
        data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker' : [smoker],
            'region' : [region]
        })
        output = predict(data)
        st.info('This is only an indicative amount, actuals may vary.', icon="ℹ️")
        st.metric(label="output", value=output,label_visibility="hidden")