import torch


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
