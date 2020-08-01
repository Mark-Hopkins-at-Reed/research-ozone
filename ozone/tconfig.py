import torch
from ozone.networks import TiedClassifier
   
class TrainingConfig:

    def __init__(self):
        self.hyperparams = dict()
        self.hyperparams['batchsize'] = 128
        self.hyperparams['optimizer'] = {'name': 'sgd',
                                         'rate': 0.001,
                                         'momentum': 0.9}
        self.hyperparams['network'] = {'name': 'tied',
                                       'numchoices': 5,
                                       'numlayers': 2,
                                       'hiddensize': 200,
                                       'dropout': {'prob': 0.2},
                                       'layernorm': False,
                                       }
        self.network_names = {'tied': TiedClassifier}
    
    
    def __getitem__(self, hparam):
        return self.hyperparams[hparam]

    def get_batch_size(self):
        return self.hyperparams['batchsize']

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
            result = network_class.create_factory_method(self.hyperparams['network'])
        else:
            result = None
        return result
    
       
    def replace(self, param_name, value):
        result = TrainingConfig()
        result.hyperparams = self.hyperparams.copy()
        result.hyperparams[param_name] = value
        return result
 

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


