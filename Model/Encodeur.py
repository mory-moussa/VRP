from torch import nn

"""L'encodeur est utilisé pour encoder les états statiques et dynamiques d'un problème de VRP."""


class Encodeur(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Encodeur, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output
