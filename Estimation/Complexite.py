import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""utilisée pour initialiser les couches de convolution du module de critique"""


class Critique(nn.Module):

    def __init__(self, hidden_size):
        super(Critique, self).__init__()

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):

        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output
