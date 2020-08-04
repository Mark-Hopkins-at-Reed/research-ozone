    
import nltk
import random
nltk.download('wordnet')

from nltk.corpus import wordnet as wn

def synsets_of_mercury():
    synsets = wn.synsets('mercury')
    for synset in synsets:
        print(synset)
        

def hypernym_chain(synset_name):
    """
    e.g. hypernym_chain('boat.n.01')
    
    """
    synset = wn.synset(synset_name)
    result = [synset]
    while len(synset.hypernyms()) > 0:
        synset = synset.hypernyms()[0]
        result.append(synset)
    return result

def get_all_hypernyms_from_sense(sense):
    result = set()
    for y in sense.hypernyms():
        result.add(y)
        for z in get_all_hypernyms_from_sense(y):
            result.add(z)
    return result

def get_all_hypernyms(word):
    """
    e.g. get_all_hypernyms('dog')
    
    """
    result = set()
    for x in wn.synsets(word):
        for y in get_all_hypernyms_from_sense(x):
            result.add(y)
    return result

def get_hyponyms(synset_name):
    """
    e.g. get_all_hyponyms('metallic_element.n.01')
    
    """
    sense = wn.synset(synset_name)
    return sense.hyponyms()


def get_all_hyponyms_from_sense(sense):
    """
    e.g. get_all_hyponyms_from_sense(wn.synset('metallic_element.n.01'))
    
    """
    result = set()
    for y in sense.hyponyms():
        result.add(y)
        for z in get_all_hyponyms_from_sense(y):
            result.add(z)
    return result

def normalize_lemma(lemma):
    lemma = " ".join(lemma.split("_"))
    lemma = " ".join(lemma.split("-"))
    lemma = lemma.lower()
    return lemma

def get_all_lemmas_from_sense(sense):
    result = set()
    for lemma in sense.lemmas():
        result.add(normalize_lemma(lemma.name()))
    for y in get_all_hyponyms_from_sense(sense):
        for lemma in y.lemmas():
            result.add(normalize_lemma(lemma.name()))
    return result



class Specificity:
    def __init__(self):
        self.cache = dict()
        
    def evaluate(self, sense):
        if sense.name() not in self.cache:
            spec = len(get_all_lemmas_from_sense(sense))
            self.cache[sense.name()] = spec
        return self.cache[sense.name()]


specificity = Specificity()


def find_lowest_common_ancestor(words):
    """
    find_lowest_common_ancestor(['libertarian', 'green', 'garden', 'democratic'])
    find_lowest_common_ancestor(['apple', 'banana', 'orange', 'grape'])
    
    """
    commonHypernyms = get_all_hypernyms(words[0])
    for word in words[1:]:
        commonHypernyms = commonHypernyms & get_all_hypernyms(word)
    if len(commonHypernyms) == 0:
        hyp = wn.synset('entity.n.01')
        return (specificity.evaluate(hyp), hyp)
    scoredHypernyms = [(specificity.evaluate(hyp), hyp) for hyp in commonHypernyms]
    sortedHypernyms = sorted(scoredHypernyms)
    return sortedHypernyms[0]

class GetRandomSynset:
    def __init__(self, root_synset):
        entity = wn.synset(root_synset)
        self.entity_hyps = get_all_hyponyms_from_sense(entity)
        specificity = Specificity()
        self.specificities = {hyponym: specificity.evaluate(hyponym) for 
                              hyponym in self.entity_hyps}
        

    def __call__(self):
        random_word = random.sample(self.entity_hyps,1)[0]
        return random_word
    
    @staticmethod    
    def factory(root_synset):
        return GetRandomSynset(root_synset)
    

    def random_synset_with_specificity(self, lower, upper):
        candidates = [hyp for hyp in self.specificities if 
                      self.specificities[hyp] >= lower and
                      self.specificities[hyp] <= upper]
        if len(candidates) > 0:
            return random.choice(candidates)
        else:
            return None

    def random_non_hyponym(self, synset_name):
        synset = wn.synset(synset_name)
        hyponyms = get_all_hyponyms_from_sense(synset) | set([synset])
        other_synsets = list(self.entity_hyps - hyponyms)
        assert(len(other_synsets) > 0)
        return random.choice(other_synsets)



def show_puzzles(puzzles):
    score = 0
    num_puzzles_seen = 0
    lives = 100
    #"puzzle" variableis unshuffled, answer is always at puzzle[4] 
    for puzzle in puzzles:
        num_puzzles_seen += 1 
        if lives == 0:
            print(score, num_puzzles_seen)
            print("GAME OVER! you got ",100 * score / num_puzzles_seen , "% correct")
            return 0
        shuffled_puzzle = puzzle[:]
        random.shuffle(shuffled_puzzle)
        print("\n \nPUZZLE: ",)
        for word in shuffled_puzzle:
            print(word.name()[:-5])
        print("\n you have " + str(lives) + " left.")
        
        #HUMAN PLAYER
        guess = input("Which word is the odd man out? ")
        
        #COMPUTER PLAYER 
        #guess = random.choice(puzzle).name()
        
        print("\n YOUR ANSWER: ", guess)
        print("\n CORRECT ANSWER: ", puzzle[4].name()[:-5])
        if (guess == puzzle[4].name()[:-5]):
           print("\n GOOD WORK!")
           score += 1
        else:
           print("\n INCORRECT")
           lives -= 1



if __name__ == "__main__":
    generate_synset = GetRandomSynset('dog.n.1')
    test_puzzles = generate_synset.generate_puzzles() 
    print(len(test_puzzles))
    
    