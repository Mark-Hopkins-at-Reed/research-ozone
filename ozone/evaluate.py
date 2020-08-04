import random
import torch
import numpy as np 
from torch import tensor
from ozone.train import predict, train
from ozone.puzzle import PuzzleDataLoader
from ozone.experiment import TrainingConfig, DEFAULT, BPE_CONFIG


class PuzzleEvaluater:

    def __init__(self, model, generator, file_path, num_choice):
        self.model = model 
        self.generator = generator
        self._build_puzzle(file_path, num_choice)

    def _build_puzzle(self, file_path, num_choice):
        reader = open(file_path).read().split('\n')
        puzzles = [line.split("    ")[1:] for line in reader]
        results = []
        for puzzle in puzzles:
            assert len(puzzle) == num_choice, "Input puzzle has a wrong length"
            index = np.random.permutation(num_choice)
            if self.generator.__class__.__name__ == "TaxonomyPuzzleGenerator":
                results.append((tuple([puzzle[i] for i in index]), index.tolist().index(0)))
            if self.generator.__class__.__name__ == "BpePuzzleGenerator":
                tok_puzzle = self.generator.bpe.apply([puzzle[i] for i in index])
                results.append(([word.split(" ") for word in tok_puzzle],index.tolist().index(0)))
        self.puzzles = results

    def evaluate(self):
        predictions = predict(self.model, self.generator.make_puzzle_matrix(self.puzzles))
        answers = tensor([answer for _,answer in self.puzzles])
        return sum(np.equal(predictions, answers)).item()/len(predictions)

if __name__ == '__main__':
    import sys
    file_path = sys.argv[1]
    num_choice = int(sys.argv[2])
    config = TrainingConfig(BPE_CONFIG)
    model = torch.load("best.model")
    puzzle_gen = config.create_puzzle_generator()
    evaluator = PuzzleEvaluater(model, puzzle_gen, file_path, num_choice)
    print(evaluator.evaluate())
