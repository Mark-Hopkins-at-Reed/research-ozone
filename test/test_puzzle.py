import unittest
import torch
from torch import tensor 
from ozone.puzzle import WordnetPuzzleGenerator, one_hot, make_puzzle_targets


class TestPuzzle(unittest.TestCase):

    def setUp(self):
        self.generator = WordnetPuzzleGenerator("apple.n.01", 3)

    def test_vocab(self):
        vocab = self.generator.get_vocab()
        assert vocab == {'apple': 0, 'baldwin': 1, 
                         "bramley's seedling": 2, 'cooking apple': 3, 
                         'cortland': 4, "cox's orange pippin": 5, 
                         'crab apple': 6, 'crabapple': 7, 
                         'delicious': 8, 'dessert apple': 9, 
                         'eating apple': 10, 'empire': 11, 
                         'golden delicious': 12, 'granny smith': 13, 
                         "grimes' golden": 14, 'jonathan': 15, 
                         "lane's prince albert": 16, 'macoun': 17, 
                         'mcintosh': 18, 'newtown wonder': 19, 
                         'northern spy': 20, 'pearmain': 21, 
                         'pippin': 22, 'prima': 23, 
                         'red delicious': 24, 'rome beauty': 25, 
                         'stayman': 26, 'stayman winesap': 27, 
                         'winesap': 28, 'yellow delicious': 29}

    def test_batch_generate(self):
        puzzles = self.generator.batch_generate(number_of_puzzles = 3)
        assert len(puzzles) == 3

    def test_make_puzzle_matrix(self):
        puzzles = self.generator.batch_generate(number_of_puzzles = 3)
        matrix = self.generator.make_puzzle_matrix(puzzles)
        assert matrix.size() == torch.Size([3, 90])

    def test_one_hot(self):
        vocab = self.generator.get_vocab()
        onehotVec = one_hot("empire", vocab)
        assert len(onehotVec) == 30 
        assert onehotVec == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_make_puzzle_targets(self):
        labels = [0, 2, 1, 1, 2, 1, 0, 0, 0, 1]
        targets = make_puzzle_targets(labels)
        assert targets.tolist() == labels

if __name__ == "__main__":
    unittest.main()