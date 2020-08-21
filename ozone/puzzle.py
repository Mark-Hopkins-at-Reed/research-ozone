import fastBPE
import numpy as np
from ozone.util import cudaify, FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader

class PuzzleGenerator:
    
    def __init__(self):
        pass

    def batch_generate(self, number_of_puzzles = 10):
        return [self.generate() for n in range(number_of_puzzles)]

    def generate(self):
        raise NotImplementedError('cannot call .generate() on abstract class.')

    def get_vocab(self):
        raise NotImplementedError('cannot call .get_vocab() on abstract class.')

    def num_choices(self):
        raise NotImplementedError('cannot call .num_choices() on abstract class.')
                
    def make_puzzle_matrix(self, puzzles):
        matrix = []
        for puzzle in puzzles:
            choices, _ = puzzle
            oneHotVec = []
            for choice in choices:
                oneHotVec += one_hot(str(choice), self.get_vocab())
            matrix.append(oneHotVec)
        return cudaify(FloatTensor(matrix))

    def tensorify(self, puzzles, num_choice):
        results = []
        for puzzle in puzzles:
            if len(puzzle) == int(num_choice):
                index = np.random.permutation(num_choice)
                results.append((tuple([puzzle[i] for i in index]), index.tolist().index(0)))
        return results 

def one_hot(word, vocab):
    vec = [0]*len(vocab)
    vec[vocab[word]] = 1
    return vec

def make_puzzle_vector(puzzle, vocab):
    choices, _ = puzzle
    oneHotVec = []
    for choice in choices:
        oneHotVec += one_hot(str(choice), vocab)
    return cudaify(FloatTensor(oneHotVec).view(1, -1))

def make_puzzle_target(label):
    return cudaify(LongTensor([label]))

def make_puzzle_targets(labels):
    return cudaify(LongTensor(labels))



class BpePuzzleGenerator(PuzzleGenerator):
    """
    Generate the tokenized puzzle
    
    """
    def __init__(self, base_puzzle_gen, vocab, bpe, num_tok, debugging=False):
        super(BpePuzzleGenerator, self).__init__()
        self.vocab = vocab
        self.bpe = bpe
        self.base_puzzle_gen = base_puzzle_gen
        self.num_tok = num_tok
        self.debugging = debugging
      
    def num_choices(self):
        return self.base_puzzle_gen.num_choices()
      
    @staticmethod
    def _read_vocab(vocab_file_path):
        with open(vocab_file_path) as reader:
            vocab = [(line.split()[0], i) for (i, line) in enumerate(reader)]
            tok_to_ix = dict(vocab)
        return tok_to_ix

    def max_tokens_per_choice(self):
        return 5
     
    def get_vocab(self):
        return self.vocab

    def generate(self):
        '''
        e.g
        result = [([['app', 'le'], ['pea', 'r']] , 0), 
                  ([['do', 'g'], ['ca', 't']], 1), 
                  ([['low', 'er'], ['high', 'er']] 0)]
        '''
        puzzle = self.base_puzzle_gen.generate()
        tok_puzzle = self.bpe.apply(list(puzzle[0]))
        new_puzzle = ([word.split(" ") for word in tok_puzzle], puzzle[1])
        return new_puzzle

    def reset_root(self, root_synset):
        self.base_puzzle_gen.reset_root(root_synset)

    def make_puzzle_matrix(self, tok_puzzles):
        '''
        concatenate first 4 tokens if exist, then merge the rest tokens 
        and append it to the end
        
        TODO: Is it possible to get rid of the topmost for-loop using torch tensor ops??
        
        '''
        matrix = []
        for tok_puzzle in tok_puzzles:
            choices, _ = tok_puzzle
            oneHotVec = []
            for choice in choices:
                choice_Vec_list = [one_hot(tok, self.vocab) for tok in choice]
                if len(choice_Vec_list) > (self.num_tok - 1):
                    choice_Vec_list[self.num_tok - 1] = [sum(vec) for vec in zip(*choice_Vec_list[self.num_tok-1:])]
                    choice_Vec_list = choice_Vec_list[:self.num_tok]
                result = [tok for word in choice_Vec_list for tok in word]
                appendix = [0] * (self.num_tok*len(self.vocab) - len(result))
                oneHotVec += result + appendix 
            matrix.append(oneHotVec)
        result = cudaify(FloatTensor(matrix))
        return result 
    
    def tensorify(self, puzzles, num_choice):
        results = []
        for puzzle in puzzles:
            assert len(puzzle) == int(num_choice)
            index = np.array(list(range(num_choice)))
            if not self.debugging:
                index = np.random.permutation(num_choice)
            randomized = [puzzle[i] for i in index]
            print(randomized)
            tok_puzzle = self.bpe.apply(randomized)
            results.append(([word.split(" ") for word in tok_puzzle],index.tolist().index(0)))
        return results 

    @staticmethod
    def from_paths(base_puzzle_gen, train_file_path, vocab_file_path, num_tok):
        vocab = BpePuzzleGenerator._read_vocab(vocab_file_path)
        bpe = fastBPE.fastBPE(train_file_path, vocab_file_path)
        return BpePuzzleGenerator(base_puzzle_gen, vocab, bpe, num_tok)
    
class PuzzleDataset(Dataset):

    def __init__(self, puzzle_generator, num_train):
        self.num_choices = puzzle_generator.num_choices()
        puzzles = puzzle_generator.batch_generate(num_train)
        self.puzzle_generator = puzzle_generator
        self.response_vector = make_puzzle_targets([label for (_, label) in puzzles])
        self.evidence_matrix = self.puzzle_generator.make_puzzle_matrix(puzzles)
        self.vocab = puzzle_generator.get_vocab()
        
    def input_size(self):
        input_size = (len(self.vocab) * 
                      self.num_choices * 
                      self.puzzle_generator.max_tokens_per_choice())
        return input_size

    def output_size(self):
        return self.puzzle_generator.num_choices()

    def __getitem__(self, index):
        return self.evidence_matrix[index], self.response_vector[index]

    def __len__(self):
        return len(self.evidence_matrix)   

         
    @staticmethod
    def compile_puzzle(generator, puzzle):
        return generator.make_puzzle_matrix([(puzzle, -1)])         

class PuzzleDataLoader:
    
    def __init__(self, puzzle_generator, num_train, train_batch_size, 
                 num_test = 1000, test_batch_size=100):
        self.puzzle_generator = puzzle_generator
        self.num_train = num_train
        self.train_batch_size = train_batch_size
        self.num_test = num_test
        self.test_batch_size = test_batch_size
        self._regenerate()

    def _regenerate(self):
        self.train_data = PuzzleDataset(self.puzzle_generator, self.num_train)
        self.train_loader = DataLoader(dataset = self.train_data, 
                                       batch_size = self.train_batch_size, 
                                       shuffle=True)
        self.test_data = PuzzleDataset(self.puzzle_generator, self.num_test)
        self.test_loader = DataLoader(dataset = self.test_data, 
                                      batch_size = self.test_batch_size, 
                                      shuffle=False)

    def get_loaders(self, epoch):
        if epoch % 100 == 0:
            self._regenerate()
        return self.train_loader, self.test_loader     

    def input_size(self):
        return self.train_data.input_size()

    def output_size(self):
        return self.train_data.output_size()


