import unittest
from ozone.puzzle import * 
from ozone.fastbpe import *
from torch import Tensor

class TestFastBpe(unittest.TestCase):

    def setUp(self):
        vocab = {"twelve":0, 
                 "thirteen":1, 
                 "fifteen":2, 
                 "twenty":3, 
                 "nineteenth": 4}
        puzzles = [(("twelve", "thirteen", "fifteen",
                     "twenty", "nineteenth"),4)]
        codes_path = "test/data/codes_10k"
        vocab_path = "test/data/vocab_10k.txt"
        self.bpe = bpeGenerator(vocab, puzzles, codes_path, vocab_path)
        
    def test_new_puzzles(self):
        self.tok_puzzles = self.bpe.generate()
        assert self.tok_puzzles == [([['twel@@', 've'], 
                                   ['thir@@', 'teen'], 
                                   ['fif@@', 'teen'], 
                                   ['twenty'], 
                                   ['nin@@', 'et@@', 'e@@', 'enth']], 4)]

    def test_new_vocab(self):
        self.new_vocab = self.bpe.get_new_vocab()
        assert self.new_vocab == {'enth': 0, 
                                  'fif@@': 1, 
                                  'twel@@': 2, 
                                  'twenty': 3, 
                                  'e@@': 4, 
                                  'teen': 5, 
                                  'thir@@': 6, 
                                  've': 7, 
                                  'nin@@': 8, 
                                  'et@@': 9}

    def test_make_matrix(self):
        matrix = make_tok_puzzle_matrix(self.tok_puzzles, self.new_vocab)
        assert matrix == Tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])


if __name__ == "__main__":
    unittest.main()