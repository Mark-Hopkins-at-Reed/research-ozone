import random
import numpy as np
from ozone.wordnet import GetRandomSynset
from ozone.wordnet import get_all_lemmas_from_sense
from ozone.puzzle import PuzzleGenerator
from nltk.corpus import wordnet as wn


class Taxonomy:
    
    def get_vocab(self):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def random_node(self, specificity_lb, specificity_ub):
        raise NotImplementedError("Cannot call this on an abstract class.")

    def random_hyponyms(self, node, k):
        raise NotImplementedError("Cannot call this on an abstract class.")
        
    def random_non_hyponym(self, node):
        raise NotImplementedError("Cannot call this on an abstract class.")


class WordnetTaxonomy(Taxonomy):
    
    def __init__(self, root_synset_name):
        super().__init__()
        self.root_synset = wn.synset(root_synset_name)
        self.synset_gen = GetRandomSynset(root_synset_name)
        self.vocab = self._build_vocab()
        
    def get_vocab(self):
        return self.vocab

    def _build_vocab(self):
        words = sorted(list(get_all_lemmas_from_sense(self.root_synset)))
        word_to_ix = dict([(v, k) for (k,v) in enumerate(words)])
        print("vocab size: {}".format(len(word_to_ix)))
        return word_to_ix

    def random_node(self, specificity_lb, specificity_ub):
        root = self.synset_gen.random_synset_with_specificity(specificity_lb, 
                                                              specificity_ub)
        if root is None:
            return None
        while root == self.root_synset:
            root = self.synset_gen.random_synset_with_specificity(specificity_lb,
                                                                  specificity_ub)
        return root.name()
    
    def random_hyponyms(self, node, k):
        node = wn.synset(node)
        hyps = get_all_lemmas_from_sense(node) # children?
        return random.sample(hyps, k)
    
    def random_non_hyponym(self, node):
        node = wn.synset(node)
        hyps = get_all_lemmas_from_sense(node) # children?
        counter = 0
        random_hyp_lemmas = []
        while len(random_hyp_lemmas) == 0:
            if counter > 10000:
                raise Exception('Too difficult to get a non-hyponym for {}; giving up'.format(node.name()))
            random_hyp = self.synset_gen.random_non_hyponym(node.name())
            random_hyp_lemmas = set(get_all_lemmas_from_sense(random_hyp))
            random_hyp_lemmas -= hyps
            counter += 1
        random_word = random.choice(list(random_hyp_lemmas))
        return random_word

class TaxonomyPuzzleGenerator(PuzzleGenerator):
    
    def __init__(self, taxonomy, num_choices):
        super().__init__()
        self.taxonomy = taxonomy
        self.specificity_lb = 10
        self.specificity_ub = 5000
        self.n_choices = num_choices
   
    def num_choices(self):
        return self.n_choices

    def max_tokens_per_choice(self):
        return 1
    
    def get_vocab(self):
        return self.taxonomy.get_vocab()
        
    def generate(self):
        root = self.taxonomy.random_node(self.specificity_lb, 
                                         self.specificity_ub)
        puzzle = self.taxonomy.random_hyponyms(root, self.num_choices() - 1)
        random_word = self.taxonomy.random_non_hyponym(root)
        puzzle.append(random_word)
        result = [(str(choice), 0) for choice in puzzle[:-1]] + [(str(puzzle[-1]), 1)]
        random.shuffle(result)
        xyz = tuple([i for (i, _) in result])
        onehot = [j for (_, j) in result]    
        return (xyz, onehot.index(1))

    
