FROM python:3.11

RUN python3 -m pip install jupyter
RUN python3 -m pip install pyspark==3.5.1

RUN apt-get update
RUN apt-get install -y default-jre

RUN java -version

CMD python3 -m notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root 