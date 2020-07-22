import fastBPE
from ozone.puzzle import one_hot
from ozone.cuda import FloatTensor, LongTensor, cudaify

class bpeGenerator:
    """First generate tokenized puzzles, then build
       the tokenized vocab.
    """
    def __init__(self, vocab, puzzles, train_file_path, vocab_file_path):
        self.vocab = vocab
        self.new_vocab, self.bpe = self._build_new_vocab(train_file_path,
                                                         vocab_file_path)
        self.puzzles = puzzles
        self.new_puzzles = None

    def _build_new_vocab(self, train_file_path, vocab_file_path):
        bpe = fastBPE.fastBPE(train_file_path, vocab_file_path)
        tok_vocab = bpe.apply(list(self.vocab.keys()))
        tok_vocab = [word.split(" ") for word in tok_vocab]
        new_vocab = set([tok for word in tok_vocab for tok in word])
        tok_to_ix = dict([(v, k) for (k,v) in enumerate(new_vocab)])
        print("vocab size: {}".format(len(tok_to_ix)))
        return tok_to_ix, bpe 

    def get_new_vocab(self):
        return self.new_vocab

    def generate(self):
        '''
        e.g
        result = [([['app', 'le'], ['pea', 'r']] , 0), 
                  ([['do', 'g'], ['ca', 't']], 1), 
                  ([['low', 'er'], ['high', 'er']] 0)]
        '''
        result = []
        for puzzle in self.puzzles:
            tok_puzzle = self.bpe.apply(list(puzzle[0]))
            new_puzzle = [word.split(" ") for word in tok_puzzle]
            result.append((new_puzzle, puzzle[1]))
        self.new_puzzles = result 
        return result

def make_tok_puzzle_vector(tok_puzzle, tok_vocab):
    '''
    concatenate first 4 tokens if exist, then merge the rest tokens 
    and append it to the end
    '''
    choices, _ = tok_puzzle
    oneHotVec = []
    for choice in choices:
        choice_Vec_list = [one_hot(tok, tok_vocab) for tok in choice]
        if len(choice_Vec_list) > 4:
            choice_Vec_list[4] = [sum(vec) for vec in zip(*choice_Vec_list[4:])]
            choice_Vec_list = choice_Vec_list[:5]
        result = [tok for word in choice_Vec_list for tok in word]
        appendix = [0] * (5*len(tok_vocab) - len(result))
        oneHotVec += result + appendix 
    return cudaify(FloatTensor(oneHotVec).view(1, -1))

def make_tok_puzzle_matrix(tok_puzzles, tok_vocab):
    matrix = []
    for tok_puzzle in tok_puzzles:
        choices, _ = tok_puzzle
        oneHotVec = []
        for choice in choices:
            choice_Vec_list = [one_hot(tok, tok_vocab) for tok in choice]
            if len(choice_Vec_list) > 4:
                choice_Vec_list[4] = [sum(vec) for vec in zip(*choice_Vec_list[4:])]
                choice_Vec_list = choice_Vec_list[:5]
            result = [tok for word in choice_Vec_list for tok in word]
            appendix = [0] * (5*len(tok_vocab) - len(result))
            oneHotVec += result + appendix 
        matrix.append(oneHotVec)
    return cudaify(FloatTensor(matrix))