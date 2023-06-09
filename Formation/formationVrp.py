import os
import torch
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader
from Taches.VRDataset import VRDataset
from Taches.Update_dynamic import update_dynamic
from Taches.Update_mask import update_mask
from Taches.Render import render
from Taches.Recompense import reward
from Formation.formationPrincipale import entrainment
from Taches.Validateur import validateur
from Model.DRL4TSP import DRL4TSP
from Estimation.EtatComplexite import EtatCritique

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def formation_vrp(args):
    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    DICTIONNAIRE_CHARGE = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 2  # (charge , demande)

    chargeMax = DICTIONNAIRE_CHARGE[args.num_nodes]

    print('***********Debut de la formation du problem VRP*************')

    donnee_entrainnement = VRDataset(args.train_size, args.num_nodes, chargeMax, MAX_DEMAND, args.seed)

    print('***********Les donnees entremement*************',donnee_entrainnement)

    donnee_valide = VRDataset(args.valid_size, args.num_nodes, chargeMax, MAX_DEMAND, args.seed + 1)

    acteur = DRL4TSP(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size, update_dynamic, update_mask,
                    args.num_layers, args.dropout).to(device)

    complexite = EtatCritique(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    parametres = vars(args)
    parametres['train_data'] = donnee_entrainnement
    parametres['valid_data'] = donnee_valide
    parametres['reward_fn'] = reward
    parametres['render_fn'] = render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        acteur.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        complexite.load_state_dict(torch.load(path, device))

    if not args.test:
        entrainment(acteur, complexite, **parametres)

    test_data = VRDataset(args.valid_size, args.num_nodes, chargeMax, MAX_DEMAND, args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validateur(test_loader, acteur, reward, render, test_dir, num_plot=5)

    print('Durée moyenne de la tournée: ', out)
