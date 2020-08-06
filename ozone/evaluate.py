import torch
import numpy as np 
from torch import tensor
from ozone.train import evaluate
from torch.utils.data import Dataset, DataLoader
from ozone.puzzle import PuzzleDataset, make_puzzle_targets
from ozone.experiment import TrainingConfig, BPE_CONFIG

class TestDataset(PuzzleDataset):
    def __init__(self, puzzle_generator, num_choice, test_file):
        self.num_choices = puzzle_generator.num_choices()
        self.puzzle_generator = puzzle_generator
        puzzles = self._build_puzzle(test_file, num_choice)
        self.response_vector = make_puzzle_targets([label for (_, label) in puzzles])
        self.evidence_matrix = self.puzzle_generator.make_puzzle_matrix(puzzles)
        self.vocab = puzzle_generator.get_vocab()
    
    def _build_puzzle(self, file_path, num_choice):
        with open(file_path, 'r') as reader:
            puzzles = []
            for line in reader:
                line = line.lower()
                new = line.replace('-', ' ')
                new = new.replace('\n', '')
                puzzles.append(new.split("\t")[1:])             
        return self.puzzle_generator.build_from_file(puzzles, num_choice)
    
class TestDataloader:
    
    def __init__(self, test_dataset):
        self.train_data = test_dataset
        self.train_loader = DataLoader(dataset = self.train_data, 
                                       shuffle=False)
    def _regenerate(self):
        pass
    
    def get_loaders(self):
        return self.train_loader, None
        
        


if __name__ == '__main__':
    import sys
    test_file = sys.argv[1]
    num_choice = int(sys.argv[2])
    model = torch.load("best.model")
    config = TrainingConfig(BPE_CONFIG)
    puzzle_gen = config.create_puzzle_generator()
    test_dataset = TestDataset(puzzle_gen, num_choice, test_file)
    test_dataloader = TestDataloader(test_dataset).get_loaders()[0]
    print(evaluate(model, test_dataloader))
