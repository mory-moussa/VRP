import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""Cette classe de module de réseau de neurones est appelée "Attention", 
    et elle est utilisée pour calculer l'attention sur les nœuds d'entrée donnés l'état courant."""


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size), device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)
        return attns
