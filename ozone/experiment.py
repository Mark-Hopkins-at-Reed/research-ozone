import torch
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
from ozone.puzzle import BpePuzzleGenerator, PuzzleDataLoader
from ozone.taxonomy import TaxonomyPuzzleGenerator, WordnetTaxonomy
from ozone.train import train
from ozone.networks import TiedClassifier
   
DEFAULT = {'puzzle': {'genre': 'wordnet', 'root': 'dog.n.01', 'numchoices': 5},
           'tokenizer': {'name': 'simple'},
           'batchsize': 128, 
           'optimizer': {'name': 'sgd', 'rate': 0.001, 'momentum': 0.9}, 
           'network': {'name': 'tied', 
                       'hiddensize': 200, 
                       'dropout': {'prob': 0.2}}
           }

BPE_CONFIG = {'puzzle': {'genre': 'wordnet', 'root': 'living_thing.n.01', 'numchoices': 5},
              'tokenizer': {'name': 'bpe', 
                            'codes': 'data/codes_10k', 
                            'vocab': 'data/vocab_10k.txt', 
                            'num_tokens': 5},
              'batchsize': 128, 
              'optimizer': {'name': 'sgd', 'rate': 0.001, 'momentum': 0.9}, 
              'network': {'name': 'tied', 
                          'hiddensize': 200, 
                          'dropout': {'prob': 0.2}}
              }

class TrainingConfig:

    def __init__(self, hyperparams = BPE_CONFIG):
        self.hyperparams = hyperparams
        self.network_names = {'tied': TiedClassifier}
    
    
    def __getitem__(self, hparam):
        return self.hyperparams[hparam]

    def get_batch_size(self):
        return self.hyperparams['batchsize']
    
    def get_num_choices(self):
        return self.hyperparams['puzzle']['numchoices']

    def get_network_config(self):
        return self.hyperparams['network']
    
    def create_puzzle_generator(self):
        genre = self.hyperparams['puzzle']['genre']
        generator = None
        if genre == 'wordnet':
            num_choices = self.hyperparams['puzzle']['numchoices']
            taxonomy = WordnetTaxonomy(self.hyperparams['puzzle']['root'])
            generator = TaxonomyPuzzleGenerator(taxonomy, num_choices)
        else:
            raise Exception("Unrecognized puzzle genre: {}".format(genre))
        tparams = self.hyperparams['tokenizer']
        if tparams['name'] == 'bpe':
            codes_path = tparams['codes']
            vocab_path = tparams['vocab']
            num_tokens = tparams['num_tokens']
            generator = BpePuzzleGenerator.from_paths(generator, codes_path, 
                                                      vocab_path, num_tokens)              
        return generator
    
    def create_optimizer_factory(self):
        optim_params = self.hyperparams['optimizer']
        optim_name = optim_params['name']
        if optim_name == 'adam':
            factory = lambda params: torch.optim.Adam(params, 
                                                      lr=optim_params['rate'])
        elif optim_name == 'sgd':
            factory = lambda params: torch.optim.SGD(params, 
                                                     lr=optim_params['rate'],
                                                     momentum=optim_params['momentum'])
        else:
            raise Exception("Unsupported optimizer name: {}".format(optim_name))
        return factory
        
    def create_network_factory(self):
        network_name = self.hyperparams['network']['name']
        if network_name in self.network_names:
            network_class = self.network_names[network_name]
            result = network_class.create_factory_method(self)
        else:
            result = None
        return result
    
    def replace(self, param_name, value):
        result = TrainingConfig()
        result.hyperparams = self.hyperparams.copy()
        result.hyperparams[param_name] = value
        return result

    @staticmethod
    def from_json(json_file):
        with open(json_file) as reader:
            data = json.load(reader)
            configs = [TrainingConfig(params) for params in data]
        return configs
                 

def vary_hyperparam(config, top_level_param, candidates, updater): 
    orig_subdict = config[top_level_param].copy()
    varied = [orig_subdict.copy() for x in candidates]
    for i, candidate in enumerate(candidates):
        updater(varied[i], candidate)
    return [config.replace(top_level_param, net) for net in varied]

def vary_hidden_size(config, candidates):
    return vary_hyperparam(config, 'network', candidates,
                           lambda x, y: x.update({'hiddensize': y}))

def vary_dropout_prob(config, candidates):
    return vary_hyperparam(config, 'network', candidates,
                           lambda x, y: x.update({'dropout': {'prob': y}}))

def vary_num_layers(config, candidates):
    return vary_hyperparam(config, 'network', candidates,
                           lambda x, y: x.update({'numlayers': y}))
    
def vary_learning_rate(config, candidates):
    return vary_hyperparam(config, 'optimizer', candidates,
                           lambda x, y: x.update({'rate': y}))




def experiment(config):
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
    puzzle_gen = config.create_puzzle_generator()
    loader = PuzzleDataLoader(puzzle_gen, num_train=2000, 
                              train_batch_size=config.get_batch_size(), 
                              num_test = 1000, test_batch_size=100)
    _, scores = train(num_epochs=3000000,
                      config=config,                      
                      data_loader = loader,
                      multigpu=False)
    return scores

def run_experiments(configs):
    results = []
    data = []
    for config in configs:
        print(config.hyperparams)
        trajectory = experiment(config)
        x = [point[0] for point in trajectory]
        y = [point[1] for point in trajectory]
        results.append(x)
        results.append(y)
        data.append({'time': str(datetime.now()), 
                     'config': config.hyperparams, 
                     'x': x, 'y': y})
        plt.plot(*results)
    return data

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
    run_experiments(filename, configs, puzzle_gen)
    
if __name__ == '__main__':
    import sys
    try:
        os.mkdir('models')
    except:
        pass
    filename = sys.argv[1] 
    output_file = sys.argv[2]
    configs = TrainingConfig.from_json(filename)
    results = run_experiments(configs)
    with open(output_file, 'w') as writer:
        writer.write(json.dumps(results, indent=4))
