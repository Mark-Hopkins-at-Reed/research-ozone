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
        output = output.clamp(min=0)
        output = self._apply_dropout(output)
        output = self.linear2(output)
        output = output.clamp(min=0)
        output = self._apply_dropout(output)
        output = self.linear3(output)
        output = output.clamp(min=0)
        output = self._apply_dropout(output)
        output = self.linear4(output)
        return output
        

class TiedClassifier(nn.Module): 

    def __init__(self, input_size, num_labels, num_choices, hidden_size, 
                 dropout_prob, phrase_encoder_constructor):
        super(TiedClassifier, self).__init__()
        self.input_size = input_size
        self.vocab_size = input_size // num_choices
        self.num_choices = num_choices
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.word_encoder = phrase_encoder_constructor(self.vocab_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.linear3 = nn.Linear(num_choices*hidden_size, hidden_size)
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
        outputs = [self.word_encoder(t[:,i*self.vocab_size:(i+1)*self.vocab_size])
                   for i in range(self.num_choices)]
        nextout = torch.cat(outputs, dim=1) 
        nextout = self.linear3(nextout).clamp(min=0)
        nextout = self._apply_dropout(nextout)
        nextout = self.linear4(nextout).clamp(min=0)
        nextout = self._apply_dropout(nextout)
        nextout = self.linear5(nextout).clamp(min=0)
        nextout = self._apply_dropout(nextout)
        nextout = self.final_layer(nextout)
        return F.log_softmax(nextout, dim=1)

    @staticmethod
    def create_factory_method(config):
        net_config = config.get_network_config()
        assert(net_config['name'] == 'tied')
        hidden_size = net_config['hiddensize']
        num_choices = config.get_num_choices()
        if 'dropout' in net_config:
            dropout_prob = net_config['dropout']['prob']
        else:
            dropout_prob = None            
        phrase_encoder_constructor = lambda vocab_size, hidden_size: PhraseEncoder(
            vocab_size, hidden_size, dropout_prob = dropout_prob)
        return lambda x, y: TiedClassifier(x, y, num_choices, hidden_size, 
                                           dropout_prob, 
                                           phrase_encoder_constructor)
