import unittest
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import json
from datetime import datetime
from ozone.puzzle import make_puzzle_targets, WordnetPuzzleGenerator
from ozone.wordnet import hypernym_chain
from ozone.tconfig import TrainingConfig, vary_hidden_size
from ozone.fastbpe import BpePuzzleGenerator
from ozone.train import PuzzleDataset

class MockModel:
    
    def __init__(self):
        pass
    
    def to(self, x):
        pass
    
    def __call__(self, input_vec):
        pass

class TestTrain(unittest.TestCase):

    def setUp(self):
        codes_path = "data/codes_10k"
        vocab_path = "data/vocab_10k.txt"
        num_train = 3
        self.base_puzzle_generator = WordnetPuzzleGenerator("apple.n.01", 3)
        self.bpe_puzzle_generator = BpePuzzleGenerator.from_path(self.base_puzzle_generator, 
                                                                 codes_path, vocab_path)
        self.base_puzzledataset = PuzzleDataset(self.base_puzzle_generator, num_train)
        self.bpe_puzzledataset = PuzzleDataset(self.bpe_puzzle_generator, num_train)


if __name__ == "__main__":
    unittest.main()