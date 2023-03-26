import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Encodeur import Encodeur
from Model.Pointeur import Pointer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Cette classe implémente un solveur de problème VRP (Vehicle Routing Problem) à l'aide d'un 
réseau de neurones profond (DRL4TSP) basé sur le renforcement."""


class DRL4TSP(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(DRL4TSP, self).__init__()

        if dynamic_size < 1:
            raise ValueError('"Assurez-vous de spécifier une valeur supérieure à 0 pour le paramètre : dynamic_size:, '
                             'même si le problème n a pas d éléments dynamiques.')

        self.update_fn = update_fn
        self.mask_fn = mask_fn

        self.static_encoder = Encodeur(static_size, hidden_size)
        self.dynamic_encoder = Encodeur(dynamic_size, hidden_size)
        self.decoder = Encodeur(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, static, dynamic, decoder_input=None, last_hh=None):

        batch_size, input_size, sequence_size = static.size()

        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)

        mask = torch.ones(batch_size, sequence_size, device=device)

        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is None else 1000

        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        for _ in range(max_steps):

            if not mask.byte().any():
                break

            decoder_hidden = self.decoder(decoder_input)

            probs, last_hh = self.pointer(static_hidden, dynamic_hidden, decoder_hidden, last_hh)
            probs = F.softmax(probs + mask.log(), dim=1)

            if self.training:
                m = torch.distributions.Categorical(probs)

                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, ptr.data)
                dynamic_hidden = self.dynamic_encoder(dynamic)

                is_done = dynamic[:, 1].sum(1).eq(0).float()
                logp = logp * (1. - is_done)

            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data).detach()

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

            decoder_input = torch.gather(static, 2,
                                         ptr.view(-1, 1, 1)
                                         .expand(-1, input_size, 1)).detach()

        tour_idx = torch.cat(tour_idx, dim=1)
        tour_logp = torch.cat(tour_logp, dim=1)

        return tour_idx, tour_logp
