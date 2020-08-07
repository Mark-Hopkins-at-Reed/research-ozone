import unittest
import torch
from torch import tensor
from ozone.taxonomy import WordnetTaxonomy, TaxonomyPuzzleGenerator
from ozone.puzzle import BpePuzzleGenerator, PuzzleDataset, PuzzleDataLoader
from ozone.puzzle import PuzzleGenerator
from ozone.train import predict, evaluate
from ozone.util import cudaify

class SimpleClassifier:
    
    def __init__(self):
        pass
    
    def to(self, x):
        pass

    def eval(self):
        pass
    
    def __call__(self, input_vec):
        result = [torch.zeros(3) for i in range(input_vec.shape[0])]
        result = torch.stack(result)
        result[:,0] = 1
        return result

class SimplePuzzleGenerator(PuzzleGenerator):
    
    def __init__(self):
        super().__init__()
        self.vocab = {'ate': 0, 'eat': 1, 'eta': 2, 'ete': 3}

    def num_choices(self):
        return 3
    
    def get_vocab(self):
        return self.vocab

    def batch_generate(self, number_of_puzzles = 10):
        return [(("eat", "eta", "ete"), 2),
                (("ate", "ete", "eta"), 0)]
        
    
    def generate(self):
        return (("eat", "eta", "ete"), 2)
    

class TestTrain(unittest.TestCase):

    def setUp(self):
        codes_path = "data/codes_10k"
        vocab_path = "data/vocab_10k.txt"
        num_train = 3
        num_tokens = 5
        taxonomy = WordnetTaxonomy("apple.n.01")
        self.base_puzzle_generator = TaxonomyPuzzleGenerator(taxonomy, 3)
        self.bpe_puzzle_generator = BpePuzzleGenerator.from_paths(self.base_puzzle_generator, 
                                                                 codes_path, vocab_path,
                                                                 num_tokens)
        self.base_puzzledataset = PuzzleDataset(self.base_puzzle_generator, num_train)
        self.bpe_puzzledataset = PuzzleDataset(self.bpe_puzzle_generator, num_train)

    def compare_tensors(self, t1, t2):
        t1 = cudaify(t1)
        t2 = cudaify(t2)
        assert t1.shape == t2.shape
        assert torch.allclose(t1, t2)

    def test_puzzle_dataset(self):
        puzzgen = SimplePuzzleGenerator()
        dataset = PuzzleDataset(puzzgen, 2)
        assert len(dataset) == 2
        datum0 = dataset[0]
        expected = (tensor([0., 1., 0., 0.,   # eat = 1
                            0., 0., 1., 0.,   # eta = 2
                            0., 0., 0., 1.]), # ete = 3
                    tensor(2))                # odd-man-out = "ete"
        self.compare_tensors(datum0[0], expected[0])
        self.compare_tensors(datum0[1], expected[1])
        datum1 = dataset[1]
        expected = (tensor([1., 0., 0., 0.,   # ate = 0
                            0., 0., 0., 1.,   # ete = 3
                            0., 0., 1., 0.]), # eta = 2
                    tensor(0))                # odd-man-out = "ate"
        self.compare_tensors(datum1[0], expected[0])
        self.compare_tensors(datum1[1], expected[1])
        
    def test_puzzle_loader(self):
        puzzgen = SimplePuzzleGenerator()
        mgr = PuzzleDataLoader(puzzgen, 2, 1, 2, 1)
        _, loader = mgr.get_loaders(epoch = 0)
        for i, (data, response) in enumerate(loader):
            if i == 0:
                expected = tensor([[0., 1., 0., 0., 
                                    0., 0., 1., 0., 
                                    0., 0., 0., 1.]])
                self.compare_tensors(data, expected)
                expected = tensor([2])
                self.compare_tensors(response, expected)
            elif i == 1:
                expected = tensor([[1., 0., 0., 0., 
                                    0., 0., 0., 1., 
                                    0., 0., 1., 0.]])
                self.compare_tensors(data, expected)
                expected = tensor([0])
                self.compare_tensors(response, expected)
            else:
                assert False, "Too many results returned by loader."
            
    def test_puzzle_loader2(self):
        puzzgen = SimplePuzzleGenerator()
        mgr = PuzzleDataLoader(puzzgen, 2, 2, 2, 2)
        _, loader = mgr.get_loaders(epoch = 0)
        for i, (data, response) in enumerate(loader):
            if i == 0:
                assert data.shape == torch.Size([2, 12])
                expected = tensor([[0., 1., 0., 0., 
                                    0., 0., 1., 0., 
                                    0., 0., 0., 1.],
                                   [1., 0., 0., 0., 
                                    0., 0., 0., 1., 
                                    0., 0., 1., 0.]])
                self.compare_tensors(data, expected)
                assert response.shape == torch.Size([2])
                expected = tensor([2, 0])                
                self.compare_tensors(response, expected)
            else:
                assert False, "Too many results returned by loader."
 
    def test_predict(self):
        model = SimpleClassifier()
        puzzgen = SimplePuzzleGenerator()
        mgr = PuzzleDataLoader(puzzgen, 2, 2, 2, 2)
        _, loader = mgr.get_loaders(epoch = 0)
        for (data, response) in loader:
            predictions = predict(model, data) 
            expected = tensor([0, 0]) # predicts choice 0 for all questionss
            self.compare_tensors(predictions, expected)

    def test_evaluate(self):
        model = SimpleClassifier()
        puzzgen = SimplePuzzleGenerator()
        mgr = PuzzleDataLoader(puzzgen, 2, 2, 2, 2)
        _, loader = mgr.get_loaders(epoch = 0)
        assert evaluate(model, loader) == 0.5 # gets the first one right, second wrong

if __name__ == "__main__":
    unittest.main()