# Installing Locally on Windows using WSL

1. Install WSL
```
wsl --install
```   
2. Update WSL to latest
```
wsl --update
```   
3. Install Ubunutu Distro
```
wsl --install -d ubuntu
```   
4. Set username and password (example : admin)
5. Switch user to root
```
sudo su
```
6. Update components
```
apt-get update
```
7. Install Python
```
sudo apt install python3-full
```
8. Create and activate python environment
```
python3 -m venv airflow_env
source airflow_env/bin/activate
```
9.Install Airflow
```
pip install apache-airflow
```
10. Create airflow dir and set home directory
```
mkdir airflow
export AIRFLOW_HOME=~/airflow
```
11. Initialise DB
```
airflow db migrate
```
12. Create User
```
airflow users create --username airflow --password airflow --role Admin -e noreply@airflow.com -f airflow -l admin
```
13. Create Folders
```
cd airflow
mkdir dags plugins
```
14. Start Webserver
```
nohup airflow webserver -p 8080 >> airflow_webserver.out &
```
15. Start Scheduler
```
nohup airflow scheduler >> airflow_scheduler.out &
```
16. Get Info abotu airflow environment
```
airflow info
airflow config list
```
17. List Dags
```
airflow dags list
```

## Additional Steps

1. Move\Copy src to ml-project (or any equivalent)
2. Update airflow.cfg
   - dags_folder
   - allowed_deserialization_classes

## Execution
```
wsl --list --verbose
wsl --distribution ubuntu-airflow --user admin
/home/airflow
source airflow_env/bin/activate
cd airflow
nohup airflow webserver -p 8080 >> airflow_webserver.out &
nohup airflow scheduler >> airflow_scheduler.out &
nohup mlflow ui >> mlflow_ui.out &
```

## Refrence Links 
[Modules Management](https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/modules_management.html)

