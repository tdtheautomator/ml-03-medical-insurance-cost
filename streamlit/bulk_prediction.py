import streamlit as st
import pandas as pd

from src.config.config_variables import PredictionPipelineConfig
from src.pipelines.prediction_pipeline import PredictPipeline

PredictionPipelineConfig = PredictionPipelineConfig()

def predict (filepath):
        topredict_df=pd.read_csv(filepath)
        call_pipeline = PredictPipeline(PredictionPipelineConfig)
        call_pipeline = PredictPipeline(PredictionPipelineConfig)
        predection_result = call_pipeline.predict(topredict_df)
        df_output=pd.DataFrame({'Predicted Charge':predection_result})
        df_final=topredict_df.merge(df_output,left_index=True,right_index=True)
        df_final.sort_values(['Predicted Charge'], ascending=[True], inplace=True)
        df_final['Predicted Charge']=df_final['Predicted Charge'].round(0)
        return df_final

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

st.title("Bulk Medical Insurance Cost Predictor")

with st.form(key="input_form"):
    uploaded_file = st.file_uploader(
    "Choose a CSV file", accept_multiple_files=False,
    )

    if st.form_submit_button("Predict"):
        if uploaded_file is not None:
            output = predict(uploaded_file)
            st.write(output)