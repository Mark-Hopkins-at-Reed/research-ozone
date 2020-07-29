import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import json
from datetime import datetime
import fastBPE
from ozone.puzzle import make_puzzle_matrix, make_puzzle_targets, WordnetPuzzleGenerator
from ozone.wordnet import hypernym_chain
from ozone.tconfig import TrainingConfig, vary_hidden_size
from ozone.fastbpe import BpePuzzleGenerator, make_tok_puzzle_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CHOICES = 3

class PuzzleDataset(Dataset):

    def __init__(self, puzzle_generator, num_train):
        self.num_choices = NUM_CHOICES
        puzzles = puzzle_generator.batch_generate(num_train)
        self.vocab = puzzle_generator.get_vocab()
        self.puzzle_generator = puzzle_generator
        self.response_vector = make_puzzle_targets([label for (_, label) in puzzles])
        if puzzle_generator.__class__.__name__ == 'BpePuzzleGenerator':
            self.evidence_matrix = make_tok_puzzle_matrix(puzzles, self.vocab)       
        else:
            self.evidence_matrix = make_puzzle_matrix(puzzles, self.vocab)

    def input_size(self):
        input_size = (len(self.vocab) * 
                      self.num_choices * 
                      self.puzzle_generator.max_tokens_per_choice())
        return input_size

    def __getitem__(self, index):
        return self.evidence_matrix[index], self.response_vector[index]

    def __len__(self):
        return len(self.evidence_matrix)   

 
    @staticmethod
    def compile_puzzle(generator, puzzle, bpe = False, 
                       t_path = None, v_path = None):
        if bpe:
            vocab = fastBPE.fastBPE([puzzle], t_path, v_path).get_vocab()
        else:
            vocab = generator.get_vocab()
        return make_puzzle_matrix([(puzzle, -1)], vocab)

    @staticmethod
    def create_data_loader(dataset, batch_size):
        dataloader = DataLoader(dataset = dataset, 
                                     batch_size = batch_size, 
                                     shuffle=True)
        return dataloader            
     
       
def evaluate(model, loader):
    """Evaluates the trained network on test data."""
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, response in loader:
            input_matrix = data.to(device)
            log_probs = model(input_matrix)
            predictions = log_probs.argmax(dim=1)
            total += predictions.shape[0]
            for i in range(predictions.shape[0]):
                if response[i].item() == predictions[i].item():
                    correct += 1
    return correct / total

def predict(model, puzzle, generator, bpe = False, 
            t_path = None, v_path = None):
    compiled = PuzzleDataset.compile_puzzle(generator, puzzle, bpe, 
                                            t_path, v_path)
    model.eval()
    input_matrix = compiled.to(device)
    model = model.to(device)
    log_probs = model(input_matrix)
    prediction = log_probs.argmax(dim=1).item()
    return prediction

def predict_k(model, generator, k, bpe = False, 
              t_path = None, v_path = None):
    model.eval()
    with torch.no_grad():
        correct = 0
        for i in range(k):
            puzzle = generator.generate()
            candidate = predict(model, puzzle[0], generator, 
                                bpe, t_path, v_path)
            if candidate == puzzle[1]:
                correct += 1
            #print(puzzle)
            #print('EXPECTED: {}'.format(puzzle[0][puzzle[1]]))
            #print('GUESSED:  {}'.format(puzzle[0][candidate]))
    print('OVERALL: {}'.format(correct/k))
        

def train(final_root_synset, initial_root_synset, num_epochs, 
          num_puzzles_to_generate, config, puzzle_generator, 
          multigpu = False):

    def maybe_regenerate(epoch, prev_loader, prev_test_loader):
        if epoch % 100 == 0:
            dataset = PuzzleDataset(puzzle_generator, num_puzzles_to_generate)
            loader = DataLoader(dataset = dataset, batch_size = config.get_batch_size(), shuffle=True)
            test_dataset = PuzzleDataset(puzzle_generator, 1000)
            test_loader = DataLoader(dataset = test_dataset, batch_size = 100, shuffle=False)
            return loader, test_loader
        else:
            return prev_loader, prev_test_loader
    
    def maybe_evaluate(model, epoch, current_root, prev_best, prev_best_acc):
        best_model = prev_best
        best_test_acc = prev_best_acc
        test_acc = None
        if epoch % 100 == 99:
            test_acc = evaluate(model, test_loader)
            print('epoch {} test ({}): {:.2f}'.format(epoch, current_root, test_acc))
            if test_acc > prev_best_acc:
                best_test_acc = test_acc
                best_model = model
                print('saving new model')
                torch.save(best_model, 'best.model')
                #torch.save(best_model.state_dict(), 'best.model')
                #predict_k(model, puzzle_generator, 1000)
        return best_model, best_test_acc, test_acc
    
    def maybe_report_time():
        if False and epoch % 100 == 0 and epoch > 0:
            finish_time = time.clock()
            time_per_epoch = (finish_time - start_time) / epoch
            print('Average time per epoch: {:.2} sec'.format(time_per_epoch))

    start_time = time.clock()
    puzzle_generator.specificity_lb = 10
    input_size = (NUM_CHOICES * 
                  len(puzzle_generator.get_vocab()) * 
                  puzzle_generator.max_tokens_per_choice())
    output_size = NUM_CHOICES
    net_factory = config.create_network_factory()
    model = net_factory(input_size, output_size)
    if multigpu and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    
    model.to(device)
    loader = None
    test_loader = None
    loss_function = nn.NLLLoss()
    optimizer = config.create_optimizer_factory()(model.parameters())
    best_model = None
    best_test_acc = -1.0
    puzzle_generator.reset_root(initial_root_synset)

    scores = []    
    for epoch in range(num_epochs):
        model.train()
        model.zero_grad()
        loader, test_loader = maybe_regenerate(epoch, loader, test_loader)
        for data, response in loader:
            input_matrix = data.to(device)
            log_probs = model(input_matrix)
            loss = loss_function(log_probs, response)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        best_model, best_test_acc, test_acc = maybe_evaluate(model, epoch, 
                                                             initial_root_synset,
                                                             best_model, 
                                                             best_test_acc)
        print(test_acc)
        if test_acc is not None:
            scores.append((epoch, test_acc))
        if best_test_acc > .9 and initial_root_synset != final_root_synset:
            current_root = initial_root_synset
            initial_root_synset = hypernym_chain(initial_root_synset)[1].name()
            puzzle_generator.reset_root(initial_root_synset)
            print("Successful training of {}! Moving on to {}.".format(current_root, initial_root_synset))
            print('saving new model')
            torch.save(model, 'models/best.model')
            best_test_acc = -1.0
            loader, test_loader = maybe_regenerate(puzzle_generator, 100, 
                                                   loader, test_loader)
        if best_test_acc >= .98 and initial_root_synset == final_root_synset:
            break
        maybe_report_time()
    return best_model, scores

def experiment(config, initial_root_synset, final_root_synset, puzzle_gen):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()    
    _, scores = train(final_root_synset = final_root_synset, 
                      initial_root_synset = initial_root_synset,
                      num_epochs=3000000,
                      num_puzzles_to_generate=2000,
                      config=config,                      
                      puzzle_generator = puzzle_gen,
                      multigpu=False)
    return scores


        
        

def run_multiple(experiment_log, configs, puzzle_gen):
    assert(experiment_log.endswith('.exp.json'))    
    log_directory, log_file = os.path.split(experiment_log)
    root_synset = log_file[:-9]
    if '._until_.' in root_synset:
        initial, final = root_synset.split('._until_.')
    else:
        initial = root_synset
        final = root_synset
    results = []
    for config in configs:
        print(config.hyperparams)
        trajectory = experiment(config, initial, final, puzzle_gen)
        x = [point[0] for point in trajectory]
        y = [point[1] for point in trajectory]
        results.append(x)
        results.append(y)
        try:
            with open(experiment_log) as reader:
                data = json.load(reader)
        except FileNotFoundError:
            data = []
        with open(experiment_log, 'w') as writer:
            data.append({'time': str(datetime.now()), 
                         'config': config.hyperparams, 
                         'x': x, 'y': y})
            writer.write(json.dumps(data, indent=4))
        plt.plot(*results)

def graph_results(experiment_log):
    with open(experiment_log) as reader:
        data = json.load(reader)
    data = sorted(data, key = lambda ex: -max(ex['y']))
    results = []
    for i, experiment in enumerate(data):
        if i < 5:
            results.append(experiment['x'])
            results.append(experiment['y'])
            print(experiment['config'])
    plt.plot(*results)

def best_experiments(experiment_log, k=1):
    with open(experiment_log) as reader:
        data = json.load(reader)
    results = sorted([(-max(exp['y']), exp) for exp in data])
    return results[:k]    

def example_experiment(filename, puzzle_gen):
    sgd_config = TrainingConfig() 
    configs = vary_hidden_size(sgd_config, [100, 200, 400, 800])
    run_multiple(filename, configs, puzzle_gen)

def baseline_experiment(filename, puzzle_gen):
    sgd_config = TrainingConfig() 
    configs = [sgd_config]
    run_multiple(filename, configs, puzzle_gen)
    
if __name__ == '__main__':
    import sys
    try:
        os.mkdir('models')
    except:
        pass
    filename = sys.argv[1]
    try:
        bpe = sys.argv[2]
        train_file_path = sys.argv[3]
        vocab_file_path = sys.argv[4]
    except IndexError:
        bpe = "notbpe"

    assert(filename.endswith('.exp.json'))    
    log_directory, log_file = os.path.split(filename)
    root_synset = log_file[:-9]   
    base_puzzle_generator = WordnetPuzzleGenerator(root_synset, NUM_CHOICES)

    if bpe == "bpe":
        baseline_experiment(filename, BpePuzzleGenerator.from_paths(base_puzzle_generator, 
                                                              train_file_path, 
                                                              vocab_file_path))
    else:   
        baseline_experiment(filename, base_puzzle_generator)
    