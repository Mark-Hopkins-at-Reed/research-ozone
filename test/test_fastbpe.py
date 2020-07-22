import unittest
from ozone.fastbpe import make_tok_puzzle_vector, make_tok_puzzle_matrix 
from ozone.fastbpe import BpeGenerator
import torch

class TestFastBpe(unittest.TestCase):

    def setUp(self):
        puzzles = [(("eat", "ate", "ete",
                     "tea", "tee"), 2)]
        codes_path = "test/data/small.codes"
        vocab_path = "test/data/small.vocab"
        self.bpe = BpeGenerator(puzzles, codes_path, vocab_path)
        
    def test_new_puzzles(self):
        self.tok_puzzles = self.bpe.generate()
        assert len(self.tok_puzzles) == 1
        assert self.tok_puzzles[0] == ([['e@@', 'a@@', 't'], 
                                        ['a@@', 'te'], 
                                        ['e@@', 'te'], 
                                        ['te@@', 'a'], 
                                        ['te@@', 'e']], 2)


    def test_get_vocab(self):
        vocab = self.bpe.get_vocab()
        assert vocab == {'a@@': 0, 'e@@': 1, 'te': 2, 'te@@': 3, 
                         'a': 4, 'e': 5, 't': 6}

    def test_make_vector(self):
        tok_puzzles = self.bpe.generate()
        vocab = self.bpe.get_vocab()
        vec = make_tok_puzzle_vector(tok_puzzles[0], vocab)
        assert vec.shape == torch.Size([1, 175])
        vec = vec.tolist()
        assert vec == [[0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 
                        0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]] 

    def test_make_matrix(self):
        tok_puzzles = self.bpe.generate()
        vocab = self.bpe.get_vocab()
        matrix = make_tok_puzzle_matrix(tok_puzzles, vocab)
        matrix = matrix.tolist()
        assert matrix == [[[0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 
                        0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]


if __name__ == "__main__":
    unittest.main()