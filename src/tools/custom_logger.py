#file used to create code for custom logging

import logging
import os
from datetime import datetime

log_file_name=f"{datetime.now().strftime('%Y_%m_%d_%H')}.log"
logs_path=os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True)

log_file_path=os.path.join(logs_path,log_file_name)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__=="__main__":
    logging.info("logging started")