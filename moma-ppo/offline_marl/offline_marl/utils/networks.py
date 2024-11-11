from torch import embedding, nn
import torch.nn.functional as F
import numpy as np
import torch

class MLPNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, sigmoid_output=False, set_final_bias=False):
        super(MLPNetwork, self).__init__()
        self.sigmoid_output = sigmoid_output
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

        if set_final_bias:
            self.fc3.weight.data.mul_(0.1)
            self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        if self.sigmoid_output == True:
            return torch.sigmoid(logits)
        else:
            return logits

class MLPNetwork4Layers(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, sigmoid_output=False, set_final_bias=False):
        super().__init__()
        self.sigmoid_output = sigmoid_output
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, num_outputs)

        if set_final_bias:
            self.fc5.weight.data.mul_(0.1)
            self.fc5.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        logits = self.fc5(x)

        if self.sigmoid_output == True:
            return torch.sigmoid(logits)
        else:
            return logits
            

class SpectralMLPNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs_with_spectral_left, num_outputs_without_spectral_middle, num_outputs_with_spectral_right, hidden_size):
        super().__init__()
        self.fc1 = nn.utils.parametrizations.spectral_norm(nn.Linear(num_inputs, hidden_size))
        self.fc2 = nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_size, hidden_size))
        self.fc3 = nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_size, hidden_size))
        self.fc4 = nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_size, hidden_size))

        self.left = nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_size, num_outputs_with_spectral_left))
        self.middle = nn.Linear(hidden_size, num_outputs_without_spectral_middle)
        self.right = nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_size, num_outputs_with_spectral_right))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        logits = torch.cat((self.left(x), self.middle(x), self.right(x)), dim=1)
        return logits

class SelfAttention(nn.Module):
    def __init__(self, num_inputs, num_outputs, seq_len):
        super().__init__()
        
        self.seq_len = seq_len
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # initial embeddings to bring everything to num_output dim
        self.embed = nn.Linear(num_inputs, num_outputs)

        # self-attention
        self.Wq = nn.Linear(num_outputs, num_outputs)
        self.Wk = nn.Linear(num_outputs, num_outputs)
        self.Wv = nn.Linear(num_outputs, num_outputs)
        self.sqrt_d = num_outputs**0.5

        # norm layer 
        self.norm = nn.LayerNorm(num_outputs)

        # final soft attention
        self.soft_key = nn.Linear(num_outputs, num_outputs, bias=False)
        self.soft_query = nn.Parameter(torch.zeros(num_outputs))
        torch.nn.init.normal_(self.soft_query)

        # positional encodings
        self._pos_encoding = nn.Parameter(torch.tensor(positional_encoding(self.seq_len, num_outputs)), requires_grad=False)
        self.positional_encodings = {1: self._pos_encoding}

    def __call__(self, inputs):
        # we expect (batch, seq_len, num_inputs)
        assert len(inputs.shape) == 3
        b_size, seq_len, num_inputs = inputs.shape  


        # initial embedding

        inputs = self.embed(inputs)

        # add the positional encodings

        if b_size not in self.positional_encodings:
            ones = torch.ones(b_size, device=inputs.device)
            repeated = torch.einsum('b,ijk->bjk', ones, self._pos_encoding)
            self.positional_encodings[b_size] = nn.Parameter(repeated, requires_grad=False)
        
        inputs += self.positional_encodings[b_size]
        
        # linear automatically batches when len(shape)==3
        Q = self.Wq(inputs)
        K = self.Wk(inputs)
        V = self.Wv(inputs)

        # compute score
        score = torch.einsum('bso, bio -> bsi', Q, K)
        score = torch.softmax(score/self.sqrt_d, dim=2)

        # self-attentions
        Z = torch.einsum('bsi, bio->bso', score, V)

        # Add and normalize
        H = self.norm(inputs + Z)

        # Soft-attention
        soft_key = torch.tanh(self.soft_key(H))
        soft_score = torch.einsum('bso, o -> bs', soft_key, self.soft_query)
        soft_score = torch.softmax(soft_score, dim=1)

        embedding = torch.einsum('bs, bso -> bo', soft_score, H)

        return embedding


def get_angles(pos, i, num_inputs):
            angle_rates = 1 / 10000**((2 * (i//2)) / np.float32(num_inputs))
            return pos * angle_rates

def positional_encoding(seq_len, num_inputs):
    angle_rads = get_angles(np.arange(seq_len)[:, np.newaxis],
                            np.arange(num_inputs)[np.newaxis, :],
                            num_inputs)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
    pos_encoding = angle_rads[np.newaxis, ...]
        
    return pos_encoding

if __name__ == '__main__':
    num_inputs = 16
    seq_len = 5
    num_outputs = 56
    network = SelfAttention(num_inputs, num_outputs, seq_len)

    data = torch.randn((126, seq_len, num_inputs))

    embeddings = network(data)

    print('lol')