import torch
import unicodedata
from ozone.train import evaluate
from torch.utils.data import DataLoader
from ozone.puzzle import PuzzleDataset, make_puzzle_targets
from ozone.experiment import TrainingConfig, BPE_CONFIG

class OddOneOutDataset:
    def __init__(self, puzzle_generator, num_choice, test_file):
        self.num_choices = puzzle_generator.num_choices()
        self.puzzle_generator = puzzle_generator
        self.puzzles = self._build_puzzle(test_file, num_choice)
        self.response_vector = make_puzzle_targets([label for (_, label) in self.puzzles])
        self.evidence_matrix = self.puzzle_generator.make_puzzle_matrix(self.puzzles)
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
        return len(self.puzzles)
    
    def _build_puzzle(self, file_path, num_choice):
        with open(file_path, 'r') as reader:
            puzzles = []
            for line in reader:
                line = line.lower()
                new = line.replace('-', ' ')
                new = new.replace('\n', '')
                new = unicodedata.normalize('NFD', new).encode('ascii', 'ignore').decode('utf8')
                puzzles.append(new.split("\t")[1:]) 
        return self.puzzle_generator.tensorify(puzzles, num_choice)
    
class OddOneOutDataloader:
    
    def __init__(self, test_dataset):
        self.train_data = test_dataset
        self.train_loader = DataLoader(dataset = self.train_data, 
                                       shuffle=False)
    def _regenerate(self):
        pass
    
    def get_loaders(self):
        return self.train_loader, None
        
        
if __name__ == '__main__':
    import sys
    test_file = sys.argv[1]
    num_choice = int(sys.argv[2])
    model = sys.argv[3]
    is_gpu = sys.argv[4]
    if is_gpu == "cpu":
        model = torch.load(model, map_location=torch.device('cpu'))
    else:
        model = torch.load(model)
    config = TrainingConfig(BPE_CONFIG)
    puzzle_gen = config.create_puzzle_generator()
    test_dataset = OddOneOutDataset(puzzle_gen, num_choice, test_file)
    test_dataloader = OddOneOutDataloader(test_dataset).get_loaders()[0]  
    model = model.eval()
    print(evaluate(model, test_dataloader))
