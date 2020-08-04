import unittest
from ozone.taxonomy import WordnetTaxonomy, TaxonomyPuzzleGenerator


class TestTaxonomy(unittest.TestCase):
    
    def setUp(self):
        self.taxonomy = WordnetTaxonomy('apple.n.1')    
     
    def test_get_vocab(self):
        expected = {'apple': 0, 'baldwin': 1, "bramley's seedling": 2, 
                    'cooking apple': 3, 'cortland': 4, 
                    "cox's orange pippin": 5, 'crab apple': 6, 'crabapple': 7, 
                    'delicious': 8, 'dessert apple': 9, 'eating apple': 10, 
                    'empire': 11, 'golden delicious': 12, 'granny smith': 13, 
                    "grimes' golden": 14, 'jonathan': 15, 
                    "lane's prince albert": 16, 'macoun': 17, 
                    'mcintosh': 18, 'newtown wonder': 19, 'northern spy': 20, 
                    'pearmain': 21, 'pippin': 22, 'prima': 23, 
                    'red delicious': 24, 'rome beauty': 25, 'stayman': 26, 
                    'stayman winesap': 27, 'winesap': 28, 
                    'yellow delicious': 29}  
        assert self.taxonomy.get_vocab() == expected
        
    def test_random_node(self):
        assert self.taxonomy.random_node(22,22) == 'eating_apple.n.01'
        assert self.taxonomy.random_node(5,5) == 'cooking_apple.n.01'
        assert self.taxonomy.random_node(6,21) == None
        
    def test_random_hyponyms(self):
        expected = set(["lane's prince albert",
                        "bramley's seedling",
                        'cooking apple',
                        'newtown wonder',
                        'rome beauty'])
        result = set(self.taxonomy.random_hyponyms('cooking_apple.n.01', 5))
        assert result == expected
        
    def test_random_non_hyponym(self):
        non_eating_apples = {"bramley's seedling",
                             'cooking apple',
                             'crab apple',
                             'crabapple',
                             "lane's prince albert",
                             'newtown wonder',
                             'rome beauty'}
        for i in range(100):
            non_hyponym = self.taxonomy.random_non_hyponym('eating_apple.n.01')
            assert non_hyponym in non_eating_apples
            
    def test_puzzle_gen(self):
        cooking_apples = set(["lane's prince albert",
                              "bramley's seedling",
                              'cooking apple',
                              'newtown wonder',
                              'rome beauty'])
        puzzgen = TaxonomyPuzzleGenerator(self.taxonomy, 4)
        puzzgen.specificity_lb = 5
        puzzgen.specificity_ub = 5
        for i in range(100):
            (choices, oddman) = puzzgen.generate()
            assert len(set(choices)) == len(choices)
            for i, choice in enumerate(choices):
                if i == oddman:
                    assert choice not in cooking_apples
                else:
                    assert choice in cooking_apples
                
if __name__ == "__main__":
	unittest.main()
