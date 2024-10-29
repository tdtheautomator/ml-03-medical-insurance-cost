import sys
import pandas as pd
from src.pipelines.prediction_pipeline import PredictPipeline
import time
from src.config.config_variables import PredictionPipelineConfig
from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging

try:
    topredict_df=pd.read_csv('./test/predict_me.csv') #update with dataset file
    PredictionPipelineConfig = PredictionPipelineConfig()
    call_pipeline = PredictPipeline(PredictionPipelineConfig)
    predection_result = call_pipeline.predict(topredict_df)
    df_output=pd.DataFrame({'Predicted Charge':predection_result})
    df_final=topredict_df.merge(df_output,left_index=True,right_index=True)
    df_final.sort_values(['Predicted Charge'], ascending=[True], inplace=True)
    df_final['Predicted Charge']=df_final['Predicted Charge'].round(0)
    filepath = f'./test/{time.strftime("%Y%m%d_%H%M%S")}_PredectedOutput.csv'
    df_final.to_csv(filepath,index=False)
    logging.info("---------------------Predicted Output---------------------")
    logging.info(f"\n {df_final.to_string(index=False)}")
except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)