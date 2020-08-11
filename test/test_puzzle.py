import unittest
import torch
from ozone.taxonomy import WordnetTaxonomy, TaxonomyPuzzleGenerator
from ozone.puzzle import one_hot, make_puzzle_targets
from ozone.puzzle import BpePuzzleGenerator


class SimplePuzzleGenerator:
    
    def __init__(self):
        self.num_choices = 3
    
    def batch_generate(self, number_of_puzzles = 10):
        return [(("eat", "ate", "ete", "tea", "tee"), 2)]
 
    def generate(self):
        return (("eat", "ate", "ete", "tea", "tee"), 2)
    

class TestPuzzle(unittest.TestCase):

    def setUp(self):
        taxonomy = WordnetTaxonomy("apple.n.01")
        self.generator = TaxonomyPuzzleGenerator(taxonomy, 3)
        codes_path = "test/data/small.codes"
        vocab_path = "test/data/small2.vocab"
        num_tokens = 5
        self.bpe = BpePuzzleGenerator.from_paths(SimplePuzzleGenerator(), 
                                                 codes_path, 
                                                 vocab_path, 
                                                 num_tokens)

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
        first_puzzle = puzzles[0]
        # puzzle should have the form:
        #    ('stayman', "lane's prince albert", 'northern spy'), 1)
        ((choice1, choice2, choice3), oddman) = first_puzzle
        assert choice1 in self.generator.get_vocab()
        assert choice2 in self.generator.get_vocab()
        assert choice3 in self.generator.get_vocab()
        assert len(set([choice1, choice2, choice3])) == 3 # all unique
        assert oddman in {0,1,2} 
        
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
    

if __name__ == "__main__":
    unittest.main()