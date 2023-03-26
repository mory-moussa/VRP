import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')

from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" La classe génère aléatoirement les données statiques et
 dynamiques pour chaque instance, qui sont ensuite utilisées pour entraîner le modèle """


class VRDataset(Dataset):
    def __init__(self, num_samples, input_size, max_load=20, max_demand=9,
                 seed=None):
        super(VRDataset, self).__init__()

        if max_load < max_demand:
            raise ValueError('La capacite doit etre superieur à la demande')

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand

        locations = torch.rand((num_samples, 2, input_size + 1))
        self.static = locations

        dynamic_shape = (num_samples, 1, input_size + 1)
        loads = torch.full(dynamic_shape, 1.)

        demands = torch.randint(1, max_demand + 1, dynamic_shape)
        demands = demands / float(max_load)

        demands[:, 0, 0] = 0
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1]






