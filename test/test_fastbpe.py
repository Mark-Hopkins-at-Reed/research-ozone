import unittest
from ozone.fastbpe import BpePuzzleGenerator
import torch


class SimplePuzzleGenerator:
    
    def batch_generate(self, number_of_puzzles = 10):
        return [(("eat", "ate", "ete", "tea", "tee"), 2)]
 
    def generate(self):
        return (("eat", "ate", "ete", "tea", "tee"), 2)


class TestFastBpe(unittest.TestCase):

    def setUp(self):
        codes_path = "test/data/small.codes"
        vocab_path = "test/data/small.vocab"
        self.bpe = BpePuzzleGenerator.from_paths(SimplePuzzleGenerator(), 
                                                 codes_path, 
                                                 vocab_path)
        
    def test_new_puzzles(self):
        self.tok_puzzles = self.bpe.batch_generate(1)
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

    def test_make_matrix(self):
        tok_puzzles = self.bpe.batch_generate(1)
        vec = self.bpe.make_puzzle_matrix(tok_puzzles)
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
    """
    def test_make_matrix(self):
        tok_puzzles = self.bpe.batch_generate(1)
        vocab = self.bpe.get_vocab()
        matrix = make_tok_puzzle_matrix(tok_puzzles, vocab)
        print(matrix)
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
    """

if __name__ == "__main__":
    unittest.main()