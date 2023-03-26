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


def update_dynamic(dynamic, chosen_idx):
    visit = chosen_idx.ne(0)
    depot = chosen_idx.eq(0)

    all_loads = dynamic[:, 0].clone()
    all_demands = dynamic[:, 1].clone()

    load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
    demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

    if visit.any():
        new_load = torch.clamp(load - demand, min=0)
        new_demand = torch.clamp(demand - load, min=0)

        visit_idx = visit.nonzero().squeeze()

        all_loads[visit_idx] = new_load[visit_idx]
        all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
        all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1)

    if depot.any():
        all_loads[depot.nonzero().squeeze()] = 1.
        all_demands[depot.nonzero().squeeze(), 0] = 0.

    tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
    return torch.tensor(tensor.data, device=dynamic.device)


def update_mask(mask, dynamic, chosen_idx=None):
    loads = dynamic.data[:, 0]
    demands = dynamic.data[:, 1]

    if demands.eq(0).all():
        return demands * 0.

    new_mask = demands.ne(0) * demands.lt(loads)

    repeat_home = chosen_idx.ne(0)

    if repeat_home.any():
        new_mask[repeat_home.nonzero(), 0] = 1.
    if torch.logical_not(repeat_home).any():
        mask_indices = torch.logical_not(repeat_home).nonzero().squeeze()
        new_mask[mask_indices, 0] = 0.0

    has_no_load = loads[:, 0].eq(0).float()
    has_no_demand = demands[:, 1:].sum(1).eq(0).float()

    combined = (has_no_load + has_no_demand).gt(0)
    if combined.any():
        new_mask[combined.nonzero(), 0] = 1.
        new_mask[combined.nonzero(), 1:] = 0.

    return new_mask.float()
