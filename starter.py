import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

list_of_dirs = [
    "config",
    "data",
    "logs",
    "notebooks",
    "outputs",
    "scripts",
    "src/components",
    "src/pipeline",
    "src/tools",
    "test"
]

list_of_files = [
    "Dockerfile",
    "requirements.txt",
    "notebooks/01.data_ingestion.ipynb",
    "notebooks/02.data_validation.ipynb",
    "notebooks/03.data_transformation.ipynb",
    "notebooks/04.model_training.ipynb",
    "notebooks/05.model_validation.ipynb",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/pipeline/__init__.py",
    "src/tools/__init__.py"
]

def create_dirs(list_of_dirs):
    for dir in list_of_dirs:
        if os.path.exists(dir):
            logging.info(f"Folder Exists: {dir}")
        else:
            os.makedirs(dir,exist_ok=True)
            logging.info(f"Folder Created: {dir}")

def create_files(list_of_files):
    for file in list_of_files:
        file_dir = os.path.exists(file)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if (not os.path.exists(file)):
            with open(file, "w") as file:
                pass
                logging.info(f"Fle Created: {file}")
        else:
            logging.info(f"File Exists: {file}")
        
create_dirs(list_of_dirs)
create_files(list_of_files)