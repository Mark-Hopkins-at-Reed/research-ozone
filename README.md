# Ozone: Embeddings from Odd-One-Out Puzzles

## To locally install the package:

    pip install -e .

## To run all unit tests:

    python -m unittest

## To run a particular unit test module (e.g. test/test_bpe.py)

    python -m unittest test.test_bpe

## Training embeddings for WordNet

From the top-most directory, run the following in a Terminal:

    mkdir results
    python ozone/experiment.py config/default.config.json results/default.trial1.json 

It should hopefully take 1500-2500 epochs to reach a test performance 
exceeding 98%, at which point training will stop. Then you can graph the
results inside of an interactive Python interpreter:

    from ozone.train import *
    graph_results('results/default.trial1.json')

## To build the Docker image

    docker build --tag mhrmm/ozone:1.1 .
    
## To run as a Docker image

Upon logging into a new Unix machine, do the following:

Install Docker:
    
    curl -sSL https://get.docker.com/ | sh    

Get the Docker image from Dockerhub:

    sudo docker pull mhrmm/ozone:1.1
    
Run the Docker image using CPU:

    sudo docker run mhrmm/ozone:1.1 config/default.config.json default.trial1.json

Run the Docker image using GPUs (if on a GPU machine):

    sudo docker run --gpus all mhrmm/ozone:1.1 config/default.config.json default.trial1.json

Each training should take between 5-20 minutes, depending on the machine. 
Before running the full script, you can check to see whether the Docker 
image will run successfully by calling it without arguments, e.g. one of 
the following two commands:

    sudo docker run mhrmm/ozone:1.1
    sudo docker run --gpus all mhrmm/ozone:1.1

This should generate the following error, which simply says that the 
argument (dog.n.01.json) is missing:

    Traceback (most recent call last):
      File "/app/ozone/train.py", line 264, in <module>
        baseline_experiment(filename)
      File "/app/ozone/train.py", line 255, in baseline_experiment
        run_multiple(filename, configs)
      File "/app/ozone/train.py", line 201, in run_multiple
        assert(experiment_log.endswith('.exp.json')) 