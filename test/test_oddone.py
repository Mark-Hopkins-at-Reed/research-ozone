import types
import unittest
from ozone.puzzle import PuzzleGenerator, BpePuzzleGenerator
from ozone.experiment import TrainingConfig, BPE_CONFIG
from ozone.oddone import OddOneOutDataset, OddOneOutDataloader

class SimplePuzzleGenerator(PuzzleGenerator):
    
    def __init__(self):
        super().__init__()
        self.vocab = {'aaa': 0, 'aea': 1, 'eaa': 2, 'eea': 3, 'aae': 4, 
                      'a e a': 5, 'bbb': 6, 'a\'s': 7}
    
    def get_vocab(self):
        return self.vocab

    def num_choices(self):
        return 5 

    def tensorify(self, puzzles, num_choice):
        # delete randomness in the test
        results = []
        for puzzle in puzzles:
            assert len(puzzle) == int(num_choice), "Input puzzle has a wrong length"
            index = list(range(5))
            results.append((tuple([puzzle[i] for i in index]), index.index(0)))
        return results 

def new_tensorify(self, puzzles, num_choice):
            # delete randomness 
            results = []
            for puzzle in puzzles:
                assert len(puzzle) == int(num_choice), "Input puzzle has a wrong length"
                index = list(range(5))
                tok_puzzle = self.bpe.apply([puzzle[i] for i in index])
                results.append(([word.split(" ") for word in tok_puzzle],index.index(0)))
            return results 

class test_oddoneoutdataloaderut(unittest.TestCase):
    
    def setUp(self):
        test_file = "test/data/test.tsv"
        s = SimplePuzzleGenerator()
        b = TrainingConfig(BPE_CONFIG).create_puzzle_generator()
        b.tensorify = types.MethodType(new_tensorify, b)
        self.ooodataset_default = OddOneOutDataset(s, 5, test_file)
        self.ooodataset_bpe = OddOneOutDataset(b, 5, test_file)

    def test_puzzles(self):
        assert self.ooodataset_bpe.puzzles == [([['a@@', 'a@@', 'a'], 
                                                 ['aea'], ['e@@', 'a@@', 'a'], 
                                                 ['e@@', 'ea'], ['a@@', 'ae']], 0), 
                                               ([['a', 'e', 'a'], ['b@@', 'b@@', 'b'], 
                                                ['aea'], ['a@@', "'s"], ['aea']], 0)]

        assert self.ooodataset_default.puzzles == [(('aaa', 'aea', 'eaa', 'eea', 'aae'), 0), 
                                                   (('a e a', 'bbb', 'aea', "a's", 'aea'), 0)]

    def test_oddoneoutdataloader(self):
        ooodataloader_default = OddOneOutDataloader(self.ooodataset_default).get_loaders()[0]
        default = []
        for data, response in ooodataloader_default:
            default.append((data, response))
        assert default[0][0].tolist() == [[1., 0., 0., 0., 0., 0., 0., 0., 
                                           0., 1., 0., 0., 0., 0., 0., 0., 
                                           0., 0., 1., 0., 0., 0., 0., 0., 
                                           0., 0., 0., 1., 0., 0., 0., 0., 
                                           0., 0., 0., 0., 1., 0., 0., 0.]]

        assert default[0][1].item() == 0 

        ooodataloader_bpe = OddOneOutDataloader(self.ooodataset_bpe).get_loaders()[0]
        bpe = []
        for data, response in ooodataloader_bpe:
            bpe.append((data, response))
        assert len(bpe[0][0][0]) == len(self.ooodataset_bpe.vocab) * 5 * 5
        assert bpe[0][1].item() == 0 


        
