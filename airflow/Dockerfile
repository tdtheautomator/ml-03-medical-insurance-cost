FROM apache/airflow:2.10.2
ENV AIRFLOW__CORE__LOAD_EXAMPLES=FALSE
RUN mkdir -p ./ml-project
COPY . ./ml-project
WORKDIR ./ml-project
RUN ls
RUN pip install --no-cache-dir "apache-airflow==2.10.2" -r requirements.txt