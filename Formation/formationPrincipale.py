import os
import time
import datetime
import numpy as np
import torch
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')

from torch.utils.data import DataLoader
from Taches.Validateur import validateur

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
""""Construit les principaux réseaux d acteurs et
 de critiques, et effectue toutes les formations."""""


def entrainment(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
                render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, **kwargs):

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = np.inf

    for epoch in range(20):

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):

            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            tour_indices, tour_logp = actor(static, dynamic, x0)

            reward = reward_fn(static, tour_indices)

            critic_est = critic(static, dynamic).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('  Lot %d/%d, reward: %2.3f, loss: %2.4f, a pris: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        valid_dir = os.path.join(save_dir, '%s' % epoch)

        mean_valid = validateur(valid_loader, actor, reward_fn, render_fn, valid_dir, num_plot=5)

        if mean_valid < best_reward:
            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs ' \
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
               np.mean(times)))
