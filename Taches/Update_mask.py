import torch


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
