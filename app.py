import os
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature 

import logging

os.environ["MLFLOW_TRACKING_URI"]= "http://ec2-65-0-105-205.ap-south-1.compute.amazonaws.com:5000"
 

logging.basicConfig(level= logging.WARN)
logger = logging.getLogger(__name__)

##evaluation metrics
def eval_metrics(actual,pred):
    rsme = np.sqrt(mean_squared_error(actual,pred))
    mse = mean_squared_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rsme,mse,r2


if __name__ =="__main__":
    
    ##data ingestion -reading the dataset --wine quality dataset
    csv_url=(
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    
    try:
        data=pd.read_csv(csv_url,sep=";")
    except Exception as e:
        logger.exception("Unable to download the data")
        
    ##spiltting the data to train and test
    
    train, test = train_test_split(data)
    train_X= train.drop(["quality"],axis =1)
    train_y = train["quality"]
    
    test_X = test.drop(["quality"],axis = 1)
    test_y = test["quality"]
    
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    with mlflow.start_run():
        lr = ElasticNet(alpha= alpha,l1_ratio=l1_ratio,random_state=42)
        lr.fit(train_X,train_y)
        
        
        predicted_qualities = lr.predict(test_X)
        (rmse,mae,r2) = eval_metrics(test_y,predicted_qualities)
        
        signature = infer_signature(train_X,train_y)

        print("elastic model (alpha is :{:f},l1_ration is : {:f})".format(alpha,l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        ## for the remote server AWS doing the setup 
        remote_server_uri = "http://ec2-65-0-105-205.ap-south-1.compute.amazonaws.com:5000/"
        mlflow.set_tracking_uri(remote_server_uri)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        if tracking_url_type_store!="file":
            mlflow.sklearn.log_model(
                lr,"model",registered_model_name="ElasticnetWineModel", signature = signature
            )
        else:
            mlflow.sklearn.log_model(lr,"model")
