import torch
import torch.nn as nn
import torch.nn.functional as F

def reset_parameters(module):
    for name, p in module.named_parameters():        
        if name.endswith('weight') and not 'layernorm' in name:
            nn.init.xavier_uniform_(p.data)
        elif name.endswith('bias'):
            p.data.zero_()


class PhraseEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_prob = 0.2):
        super(PhraseEncoder, self).__init__()
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(vocab_size, hidden_size)    
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        #self.layernorm = torch.nn.LayerNorm([hidden_size])
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        reset_parameters(self)
     
    def _apply_dropout(self, input_vec):
        output = input_vec
        if self.dropout_prob > 0:
            output = self.dropout(output)
        return output            
            
    def forward(self, input_vec):
        output = self.linear1(input_vec)
        #output = self.layernorm(output)
        output = output.clamp(min=0)
        output = self._apply_dropout(output)
        output = self.linear2(output)
        #output = self.layernorm(output)
        output = output.clamp(min=0)
        output = self._apply_dropout(output)
        output = self.linear3(output)
        #output = self.layernorm(output)
        output = output.clamp(min=0)
        output = self._apply_dropout(output)
        output = self.linear4(output)
        return output
        
class ResidualPhraseEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_prob = 0.2):
        super(ResidualPhraseEncoder, self).__init__()
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(vocab_size, hidden_size)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(2*hidden_size, hidden_size)
        reset_parameters(self)

    def _apply_dropout(self, input_vec):
        output = input_vec
        if self.dropout_prob > 0:
            output = self.dropout(output)
        return output            
 
    def forward(self, input_vec):
        output = self.linear1(input_vec).clamp(min=0)
        output = self._apply_dropout(output)
        residual = output.clone()
        output = self.linear2(output).clamp(min=0)
        output = self._apply_dropout(output)
        output_clone = output.clone()
        output = self.linear3(torch.cat([output, residual], dim=1)).clamp(min=0)
        output = self._apply_dropout(output)
        residual = output_clone
        output = self.linear4(torch.cat([output, residual], dim=1))
        return output
    
class TiedClassifier(nn.Module): 

    def __init__(self, input_size, num_labels, hidden_size, 
                 dropout_prob, phrase_encoder_constructor):
        super(TiedClassifier, self).__init__()
        self.input_size = input_size
        self.vocab_size = input_size // 5
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.word_encoder = phrase_encoder_constructor(self.vocab_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.linear3 = nn.Linear(5*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.final_layer = nn.Linear(hidden_size, num_labels)
        reset_parameters(self)

    def _apply_dropout(self, input_vec):
        output = input_vec
        if self.dropout_prob > 0:
            output = self.dropout(output)
        return output            

    def forward(self, input_vec):
        t = input_vec
        output1 = self.word_encoder(t[:,0*self.vocab_size:1*self.vocab_size])
        output2 = self.word_encoder(t[:,1*self.vocab_size:2*self.vocab_size])
        output3 = self.word_encoder(t[:,2*self.vocab_size:3*self.vocab_size])
        output4 = self.word_encoder(t[:,3*self.vocab_size:4*self.vocab_size])
        output5 = self.word_encoder(t[:,4*self.vocab_size:5*self.vocab_size])
        nextout = torch.cat([output1, output2, output3, output4, output5], dim=1) 
        nextout = self.linear3(nextout).clamp(min=0)
        nextout = self._apply_dropout(nextout)
        nextout = self.linear4(nextout).clamp(min=0)
        nextout = self._apply_dropout(nextout)
        nextout = self.linear5(nextout).clamp(min=0)
        nextout = self._apply_dropout(nextout)
        nextout = self.final_layer(nextout)
        return F.log_softmax(nextout, dim=1)
    
class ThreeWayTiedClassifier(nn.Module): 

    def __init__(self, input_size, num_labels, hidden_size, 
                 dropout_prob, phrase_encoder_constructor):
        super(ThreeWayTiedClassifier, self).__init__()
        self.input_size = input_size
        self.vocab_size = input_size // 3
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.word_encoder = phrase_encoder_constructor(self.vocab_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_prob)
        #self.layernorm = torch.nn.LayerNorm([hidden_size])
        self.linear3 = nn.Linear(3*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.final_layer = nn.Linear(hidden_size, num_labels)
        reset_parameters(self)

    def _apply_dropout(self, input_vec):
        output = input_vec
        if self.dropout_prob > 0:
            output = self.dropout(output)
        return output            

    def forward(self, input_vec):
        t = input_vec
        output1 = self.word_encoder(t[:,0*self.vocab_size:1*self.vocab_size])
        output2 = self.word_encoder(t[:,1*self.vocab_size:2*self.vocab_size])
        output3 = self.word_encoder(t[:,2*self.vocab_size:3*self.vocab_size])
        nextout = torch.cat([output1, output2, output3], dim=1) 
        nextout = self.linear3(nextout)
        #nextout = self.layernorm(nextout)
        nextout = nextout.clamp(min=0)
        nextout = self._apply_dropout(nextout)
        nextout = self.linear4(nextout)
        #nextout = self.layernorm(nextout)
        nextout = nextout.clamp(min=0)
        nextout = self._apply_dropout(nextout)
        nextout = self.linear5(nextout)
        #nextout = self.layernorm(nextout)
        nextout = nextout.clamp(min=0)
        nextout = self._apply_dropout(nextout)
        nextout = self.final_layer(nextout)
        return F.log_softmax(nextout, dim=1)


    @staticmethod
    def create_factory_method(config):
        assert(config['name'] == '3way')
        hidden_size = config['hiddensize']
        num_hidden_layers = config['numlayers']
        if 'dropout' in config:
            dropout_prob = config['dropout']['prob']
        else:
            dropout_prob = None            
        phrase_encoder_constructor = lambda vocab_size, hidden_size: PhraseEncoder(
            vocab_size, hidden_size, dropout_prob = dropout_prob)
        return lambda x, y: ThreeWayTiedClassifier(x, y, hidden_size, 
                                                  dropout_prob,
                                                  phrase_encoder_constructor)

    
"""
def initialize_net(config_name, input_size, output_size):
    params = {'arch': 'tp',
              'dp': 0.2}
    settings = config_name.split('//')
    for setting in settings:
        param_name, param_value = setting.split(':')
        params[param_name] = param_value
    
    arch = params['arch']
    if arch == 'tp':
        return TiedClassifier(input_size, output_size, 300, float(params['dp']),
                              lambda vocab_size, hidden_size: PhraseEncoder(vocab_size, hidden_size, dropout_prob = float(params['dp'])))
    elif arch == 'tr':
        return TiedClassifier(input_size, output_size, 300, float(params['dp']),
                              lambda vocab_size, hidden_size: ResidualPhraseEncoder(vocab_size, hidden_size, dropout_prob = float(params['dp'])))
    elif arch == 'tp3':
        return ThreeWayTiedClassifier(input_size, output_size, 300, float(params['dp']),
                              lambda vocab_size, hidden_size: PhraseEncoder(vocab_size, hidden_size, dropout_prob = float(params['dp'])))
    elif arch == 'tr3':
        return ThreeWayTiedClassifier(input_size, output_size, 300, float(params['dp']),
                              lambda vocab_size, hidden_size: ResidualPhraseEncoder(vocab_size, hidden_size, dropout_prob = float(params['dp'])))
    else:
        raise Exception("Unrecognized network architecture: {}"
                        .format(arch))
              
""" 
    
    
    