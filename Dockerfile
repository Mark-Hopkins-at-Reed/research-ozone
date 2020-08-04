FROM nvidia/cuda:10.2-base
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install torch==1.5.1 matplotlib==3.3.0 nltk==3.5
COPY . /app
WORKDIR /app
RUN pip3 install -e .
ENTRYPOINT ["python3", "/app/ozone/experiment.py"]
