# Medical Insurance Charge Prediction (Regression)

The project is created for learning purposes incorporating several methods.

## Project Features

1. training pipeline with 4 tasks
   - data ingestion -> data valiation --> data transformation --> model training
2. prediction pipeline, including batch prediction
3. drift detection using evidently
4. Tracking Metrics and Parameters using MLFLOW
5. Uses regressiong metrics for automatically detecting best model performance and registering it in MLFLow
6. Simple api to call training pipeline using FastAPI
7. Simple UI using streamlit for form based and batch prediction
8. Scheduling training pipeline using airflow




## Dataset
This is a simple dataset to detect medical insurance charge prediciton (without existing medical history and family histor.
Sample dataset is used form [Kaggle](https://www.kaggle.com/)<br />


 - dataset shape : Rows : 1338 , Columns : 7<br />
 - numerical features : 4 : ['age', 'bmi', 'children', 'charges']
 - categorical features : 3 : ['sex', 'smoker', 'region']
 - training dataset : 80%
 - dependent feature : charges
 - independent feature : age, bmi, children, sex, smoker, region


## Algorithms Used
1. Catagory Boost Regressor
2. Decision Tree Regresso
3. Random Forest Regressor

### Performance Metrics
1. MAE (Mean Absolute Error)
2. MSE (Mean Squared Error)
3. RMSE (Root Mean Sqaured Error)
4. R2 Score (R Squared)

## Usage (CLI)

- Ensure anaconda is installed [Anaconda Download](https://www.anaconda.com/download)
- Clone git repo
- Create new virtual environment
```
conda create -p venv python==3.12

```

- Activate new virtual environment
```
conda activate ./venv
```

- Deploy requirements
```
pip install -r requirements.txt
```

- Variables can be updated at ./src/vars/__init__.py
- Run FastAPI (default: http://127.0.0.1:8000/)
```
  python fastapi\app.py
```
- Run Streamlit (http://127.0.0.1:8081/  http://127.0.0.1:8082/)
```
streamlit run \streamlit\app.py --server.port 8081
streamlit run \streamlit\bulk_prediction.py --server.port 8082
```
- Run MLFlow (default: http://127.0.0.1:5000)
```
  mlflow ui
```


## Notes
- Detailed logging is available under ./logs
- Output files are saved in ./outputs
