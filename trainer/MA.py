import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical 
from replay_buffer import ReplayBuffer

class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )        
    
    def forward(self, obs, deterministic=False, with_logprob=True):
        x = self.actor(obs)
        return x

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )        
        self.last = nn.Linear(hidden_size, 1)
    
    def forward(self, obs, act):
        x = self.critic(torch.cat([obs, act], dim=-1))
        x = self.last(x)
        return x

class MADDPG:
    def __init__(self, obs_space, action_space, args, hidden_size=64):
        #action_dim = action_space.shape[0]
        act_dim = 1
        obs_dim = obs_space[0][0]
        self.critic = Critic(obs_dim, act_dim*len(obs_space), hidden_size=hidden_size)
        self.actor = Actor(obs_dim, hidden_size=hidden_size)
        self.replay_buffer = ReplayBuffer(1e6)
        self.alpha = 0.2
        self.gamma = 0.99
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

    def act(self, obs):
        a = self.actor(obs)
        
        dist = Categorical(a)
        action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return action, action_log_probs

    def evaluate_actions(self, obs, action):
        a = self.actor(obs)
        dist = self.dist(a)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return action, action_log_probs, dist_entropy

    def experience(self, obs, act, rew, new_obs, done):
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def update(self, agents, batch_size):
        obs_n = []
        obs_next_n = []
        act_n = []
        rew_n = []
        for i in range(len(agents)):
            obs, acts, rew, next_obs, d = agents[i].replay_buffer.sample(batch_size)
            obs_n.append(obs)
            obs_next_n.append(next_obs)
            act_n.append(act)
            rew_n.append(rew)
        
        obs, act, rew, obs_next, done = self.replay_buffer.sample(batch_size)
        rew = np.expand_dims(rew, axis=-1)
        d = np.expand_dims(done, axis=-1)        
        # compute pi loss
        action, action_log_probs = self.act(obs_n)
        q_new_actions = self.critic(obs_n, action)   
        policy_loss = (self.alpha * action_log_probs - q_new_actions).mean()

        # compute Q loss
        q_pred = self.critic(obs_n, action_n)
        next_action, next_action_log_probs = self.actor(obs_next_n)
        target_q_values = self.critic(obs_next_n, next_action) - next_action_log_probs
        q_target = rew_n + self.gamma * (1 - d) * (target_q_values - self.alpha * next_action_log_probs)
        q_loss = self.qf_criterion(q_pred, q_target.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        


# From Spinningup
class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi