import unittest
from ozone.puzzle import PuzzleGenerator, BpePuzzleGenerator
from ozone.oddone import OddOneOutDataset, OddOneOutDataloader

class SimplePuzzleGenerator(PuzzleGenerator):
    
    def __init__(self):
        super().__init__()
        self.vocab = {'aaa': 0, 'aea': 1, 'tea': 2, 'eea': 3, 'aae': 4, 
                      'a e a': 5, 'ete': 6, "t'a": 7, 'eta': 8}
    
    def get_vocab(self):
        return self.vocab

    def num_choices(self):
        return 5 
    
    def max_tokens_per_choice(self):
        return 1

    def tensorify(self, puzzles, num_choice):
        results = []
        for puzzle in puzzles:
            assert len(puzzle) == int(num_choice), "Input puzzle has a wrong length"
            index = list(range(5))
            results.append((tuple([puzzle[i] for i in index]), index.index(0)))
        return results 

class SimpleBpePuzzleGenerator:
    
    def __init__(self):
        pass
    
    def num_choices(self):
        return 5
    
    def batch_generate(self, number_of_puzzles = 10):
        return [(("eat", "ate", "ete", "tea", "tee"), 2)]
 
    def generate(self):
        return (("eat", "ate", "ete", "tea", "tee"), 2)
    


class TestOddoneDefault(unittest.TestCase):
    
    def setUp(self):
        test_file = "test/data/oddone.utf8.tsv"
        s = SimplePuzzleGenerator()
        self.dataset = OddOneOutDataset(s, 5, test_file)

    def test_puzzles(self):
        assert len(self.dataset) == 2
        (puzzle_vec, label) = self.dataset[0]
        assert label.item() == 0
        assert self.dataset.input_size() == 5 * 9
        assert self.dataset.output_size() == 5
        assert puzzle_vec.tolist() == [1., 0., 0., 0., 0., 0., 0., 0., 0., #aaa
                                       0., 1., 0., 0., 0., 0., 0., 0., 0.,#aea
                                       0., 0., 1., 0., 0., 0., 0., 0., 0.,#tea
                                       0., 0., 0., 1., 0., 0., 0., 0., 0.,#eea
                                       0., 0., 0., 0., 1., 0., 0., 0., 0.]#aae
        

    def test_dataloader(self):
        dataloader = OddOneOutDataloader(self.dataset).get_loaders()[0]
        default = []
        for data, response in dataloader:
            default.append((data, response))
        assert default[0][0].tolist() == [[1., 0., 0., 0., 0., 0., 0., 0., 0., 
                                           0., 1., 0., 0., 0., 0., 0., 0., 0.,
                                           0., 0., 1., 0., 0., 0., 0., 0., 0.,
                                           0., 0., 0., 1., 0., 0., 0., 0., 0.,
                                           0., 0., 0., 0., 1., 0., 0., 0., 0.]]

        assert default[0][1].item() == 0 

        

class TestOddoneBpe(unittest.TestCase):
    
    def setUp(self):
        test_file = "test/data/oddone.utf8.tsv"
        codes_path = "test/data/small.codes"
        vocab_path = "test/data/small.vocab"
        num_tokens = 5
        b = BpePuzzleGenerator.from_paths(SimpleBpePuzzleGenerator(), 
                                          codes_path, 
                                          vocab_path, num_tokens)
        b.debugging = True
        self.dataset = OddOneOutDataset(b, 5, test_file)
                                                 

    def test_puzzles(self):
        assert len(self.dataset) == 2
        (puzzle_vec, label) = self.dataset[0]
        assert label.item() == 0
        assert self.dataset.input_size() == 5 * 8 * 5
        assert self.dataset.output_size() == 5
        expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, #a@@
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, #a@@
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, #a
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, #null
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, #null
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert puzzle_vec.tolist() == expected 


    def test_dataloader(self):
        dataloader = OddOneOutDataloader(self.dataset).get_loaders()[0]
        default = []
        for data, response in dataloader:
            default.append((data, response))
        expected =[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, #a@@
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, #a@@
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, #a
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, #null
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, #null
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        assert default[0][0].tolist() == expected
        assert default[0][1].item() == 0 
