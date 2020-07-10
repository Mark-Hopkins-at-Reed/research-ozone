import unittest
from ozone.wordnet import find_lowest_common_ancestor, GetRandomSynset
from nltk.corpus import wordnet as wn


class TestWordnet(unittest.TestCase):
    
    def setUp(self):
        self.grs = GetRandomSynset('bird.n.1')
    
    def test_lowest_common_ancestor(self):
        ancestor = find_lowest_common_ancestor(['car','tank','bike'])
        assert ancestor == (218, wn.synset('self-propelled_vehicle.n.01'))
    
     
    def test_random_synset_with_specificity(self):
        assert self.grs.random_synset_with_specificity(120,130) == wn.synset('anseriform_bird.n.01')
 
  
       
if __name__ == "__main__":
	unittest.main()
