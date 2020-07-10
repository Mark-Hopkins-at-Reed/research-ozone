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
    python ozone/train.py results/dog.n.01.exp.json
    
It should hopefully take 1500-2500 epochs to reach a test performance 
exceeding 98%, at which point training will stop. Then you can graph the
results inside of an interactive Python interpreter:

    from ozone.train import *
    graph_results('results/dog.n.01.exp.json')