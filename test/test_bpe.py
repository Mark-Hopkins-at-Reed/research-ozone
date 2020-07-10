import unittest
from ozone.bpe import BytePairTrainer


class TestBpe(unittest.TestCase):

    def test_merge(self):
        lines = ['the dog doo', 'that dog they dogged']
        trainer = BytePairTrainer(lines)
        trainer.merge('d', 'o')
        assert trainer.tokens == [['t', 'h', 'e</w>'],
                                  ['do', 'g</w>'],
                                  ['do', 'o</w>'],
                                  [],
                                  ['t', 'h', 'a', 't</w>'],
                                  ['do', 'g</w>'],
                                  ['t', 'h', 'e', 'y</w>'],
                                  ['do', 'g', 'g', 'e', 'd</w>'],
                                  []]      
        assert trainer.pair_locations == {('t', 'h'): [0, 4, 6],
                                          ('h', 'e</w>'): [0],
                                          ('h', 'a'): [4],
                                          ('a', 't</w>'): [4],
                                          ('h', 'e'): [6],
                                          ('e', 'y</w>'): [6],
                                          ('g', 'g'): [7],
                                          ('g', 'e'): [7],
                                          ('e', 'd</w>'): [7],
                                          ('do', 'g</w>'): [1, 5],
                                          ('do', 'o</w>'): [2],
                                          ('do', 'g'): [7]}
    
    def test_train_simple(self):
        lines = ['the dog doo', 'that dog they dogged']
        trainer = BytePairTrainer(lines)
        encoder = trainer.train(3)
        print(encoder.encode(lines))
        assert(encoder.merges == [('d', 'o'), ('t', 'h'), ('do', 'g</w>')])
        assert(encoder.encode(lines) == 
               "th&& e dog do&& o\nth&& a&& t dog " +
               "th&& e&& y do&& g&& g&& e&& d\n")

    def test_train_exceed_max(self):
        lines = ['the dog doo', 'that dog they dogged']
        trainer = BytePairTrainer(lines)
        encoder = trainer.train(15)
        assert encoder.merges == [('d', 'o'), 
                                  ('t', 'h'), 
                                  ('do', 'g</w>'), 
                                  ('a', 't</w>'), 
                                  ('e', 'y</w>'), 
                                  ('g', 'g'), 
                                  ('e', 'd</w>'), 
                                  ('do', 'o</w>'), 
                                  ('th', 'e</w>'), 
                                  ('th', 'at</w>'), 
                                  ('th', 'ey</w>'), 
                                  ('do', 'gg'), 
                                  ('dogg', 'ed</w>')]
        assert encoder.encode(lines) == "the dog doo\nthat dog they dogged\n"
        
if __name__ == "__main__":
	unittest.main()