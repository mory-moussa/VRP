import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Encodeur import Encodeur

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""cette classe est utilisée pour estimer la complexité d'un problème donné."""


class StateCritic(nn.Module):

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encodeur(static_size, hidden_size)
        self.dynamic_encoder = Encodeur(dynamic_size, hidden_size)

        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output
