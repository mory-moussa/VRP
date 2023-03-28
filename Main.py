import argparse
from Formation.formationVrp import formation_vrp

if __name__ == '__main__':
    parseur = argparse.ArgumentParser(description='Optimisation Combinatoire')
    parseur.add_argument('--seed', default=12345, type=int)
    parseur.add_argument('--checkpoint', default=None)
    parseur.add_argument('--test', action='store_true', default=False)
    parseur.add_argument('--task', default='vrp')
    parseur.add_argument('--nodes', dest='num_nodes', default=10, type=int)
    parseur.add_argument('--actor_lr', default=5e-4, type=float)
    parseur.add_argument('--critic_lr', default=5e-4, type=float)
    parseur.add_argument('--max_grad_norm', default=2., type=float)
    parseur.add_argument('--batch_size', default=256, type=int)
    parseur.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parseur.add_argument('--dropout', default=0.1, type=float)
    parseur.add_argument('--layers', dest='num_layers', default=1, type=int)
    parseur.add_argument('--train-size', default=1000000, type=int)
    parseur.add_argument('--valid-size', default=1000, type=int)

    args = parseur.parse_args()

    formation_vrp(args)
