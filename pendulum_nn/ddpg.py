import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def mlp(input_dim, output_dim, hidden_dims=(128, 128), activation=nn.ReLU):
    layers = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(last_dim, h), activation()])
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ReplayBuffer:
    def __init__(self, size=100000):
        self.size = size
        self.ptr = 0
        self.full = False
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

    def add(self, s, a, r, ns):
        if len(self.states) < self.size:
            self.states.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.next_states.append(None)
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = ns
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        max_idx = self.size if self.full else len(self.states)
        idxs = np.random.randint(0, max_idx, size=batch_size)
        s = torch.tensor(np.array([self.states[i] for i in idxs]), dtype=torch.float32)
        a = torch.tensor(np.array([self.actions[i] for i in idxs]), dtype=torch.float32)
        r = torch.tensor(np.array([self.rewards[i] for i in idxs]), dtype=torch.float32).unsqueeze(1)
        ns = torch.tensor(np.array([self.next_states[i] for i in idxs]), dtype=torch.float32)
        return s, a, r, ns


class DDPGAgent:
    def __init__(self, obs_dim, act_dim, act_limit):
        self.actor = mlp(obs_dim, act_dim, hidden_dims=(128, 128), activation=nn.ReLU)
        self.actor_target = mlp(obs_dim, act_dim, hidden_dims=(128, 128), activation=nn.ReLU)
        self.critic = mlp(obs_dim + act_dim, 1, hidden_dims=(128, 128), activation=nn.ReLU)
        self.critic_target = mlp(obs_dim + act_dim, 1, hidden_dims=(128, 128), activation=nn.ReLU)
        self.act_limit = act_limit
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.update_target(1.0)

    def update_target(self, tau=0.005):
        with torch.no_grad():
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.mul_(1 - tau)
                tp.data.add_(tau * p.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - tau)
                tp.data.add_(tau * p.data)

    def select_action(self, obs, noise_scale=0.1):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        action += noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -self.act_limit, self.act_limit)

    def train(self, replay, batch_size=64, gamma=0.99):
        s, a, r, ns = replay.sample(batch_size)
        with torch.no_grad():
            next_a = self.actor_target(ns)
            q_target = self.critic_target(torch.cat([ns, next_a], dim=1))
            y = r + gamma * q_target
        q = self.critic(torch.cat([s, a], dim=1))
        critic_loss = nn.MSELoss()(q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(torch.cat([s, self.actor(s)], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target()
