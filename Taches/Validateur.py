import os
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""Elle est utilisée pour surveiller les performances du modèle sur un ensemble de validation et, 
si nécessaire, pour enregistrer des tracés de solution."""


def validateur(data_loader, actor, reward_fn, render_fn=None, save_dir='.', num_plot=5):

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png' % (batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)
