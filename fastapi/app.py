import os
import sys
import time

from src.exception.custom_exception import CustomException
from src.logging.custom_logger import logging
from src.pipelines.training_pipeline import TrainingPipeline

import certifi
ca = certifi.where()

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.start_training_pipeline()
        return Response("Training Pipeline Triggered")
    except Exception as e:
        raise CustomException(e,sys)

@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        return Response("Prediction Pipeline Triggered")
    except Exception as e:
            raise CustomException(e,sys)
    
if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=8000)