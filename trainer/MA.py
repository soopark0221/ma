import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical 
from replay_buffer import ReplayBuffer
from copy import deepcopy
from trainer.swag import SWAG
from trainer.network import Actor, Critic
import os

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class SWAGMA:
    def __init__(self, agent_n, obs_space, action_space, args, hidden_size=256):
        self.total_obs_dim = np.sum([obs_space[i].shape for i in range(len(obs_space))])
        self.total_act_dim = np.sum([action_space[i].n for i in range(len(action_space))])

        self.current_agent = agent_n
        self.act_dim = action_space[agent_n].n
        self.tau=args.tau
        self.gamma=args.gamma
        self.batch_size=args.batch_size

        self.critic = Critic(self.total_obs_dim, self.total_act_dim, hidden_size=hidden_size)
        self.actor = Actor(obs_space[agent_n].shape, action_space[agent_n].n, hidden_size=hidden_size)
        #self.critic_target = Critic(self.total_obs_dim, self.total_act_dim, hidden_size=hidden_size)
        #self.actor_target = Actor(obs_space[agent_n].shape, action_space[agent_n].n, hidden_size=hidden_size)
        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor)
        
        self.replay_buffer = ReplayBuffer(args.memory_length)
        self.noise_scale = 0.1

        self.critic_criterion = nn.MSELoss()
        self.actor_criterion = nn.MSELoss()
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.a_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.c_lr)
        self.train_step = 0

        self.swag_model = SWAG(self.critic)

        # change to cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_target.to(self.device)
        self.critic_target.to(self.device)

    
    def act(self, obs):
        # obs obs_shape 
        obs = obs.float().to(self.device)
        a = self.actor(obs.unsqueeze(0)).squeeze(0)
        a += self.noise_scale * torch.from_numpy(np.random.randn(*a.shape)).to(self.device)
        a = torch.clamp(a, -1.0, 1.0)

        a = a.detach().cpu().numpy()
        #dist = Categorical(a)
        #action = dist.sample()

        #action_log_probs = dist.log_probs(action)
        #dist_entropy = dist.entropy().mean()

        return a

    def evaluate_actions(self, obs):
        obs = obs.float()
        a = self.actor(obs)
        dist = self.dist(a)

        action_log_probs = dist.log_probs(a)
        dist_entropy = dist.entropy().mean()

        return a, action_log_probs, dist_entropy

    def experience(self, obs, act, rew, next_obs):
        self.replay_buffer.add(obs, act, rew, next_obs)

    def update(self, agents):
        # num_agent x size x batch
        obs_n = []
        next_obs_n = []
        act_n = []
        current_r = []
        for i, agent in enumerate(agents):
            o, a, r, no = agents[i].replay_buffer.sample(self.batch_size) # different idx for each agent
            if i == self.current_agent:
                current_r = r
            obs_n.append(torch.tensor(o))
            next_obs_n.append(torch.tensor(no))
            act_n.append(torch.tensor(a))


        ### compute Q loss
        all_act = []
        tensor_next_obs_n = [[] for _ in range(len(agents))]
        for i, agent in enumerate(agents):
            tensor_next_obs_n[i] = next_obs_n[i].transpose(0,1).contiguous().to(self.device) # batch_size x size
            all_act.append(agent.actor_target(tensor_next_obs_n[i].float()))
        all_act = torch.stack(all_act, dim=1).to(self.device) # batch_size x num_agent x size        
        all_act = all_act.view(self.batch_size, -1) # batch_size x (num_agent x size)   
        
        # obs
        tensor_obs_n = torch.tensor(np.vstack(obs_n)).to(self.device) # (num_agent x size) x batch_size
        tensor_obs_n = tensor_obs_n.transpose(0,1).contiguous() # batch_size x (num_agent x size)

        tensor_act_n = torch.tensor(np.vstack(act_n)).to(self.device)
        tensor_act_n = tensor_act_n.transpose(0,1).contiguous()

        # next obs
        next_obs_n = torch.tensor(np.vstack(next_obs_n)).to(self.device) # batch_size x (num_agent x size)
        next_obs_n =  next_obs_n.transpose(0,1).contiguous()
        # target Q 
        target_Q = self.critic_target(next_obs_n.float(), all_act.float())
        y = torch.tensor(current_r).to(self.device).unsqueeze(1) + self.gamma * target_Q 
        # current Q
        Q_val = self.critic(tensor_obs_n.float(), tensor_act_n.view(self.batch_size, -1))  # batch_size x size

        critic_loss = self.critic_criterion(target_Q, Q_val)

        ### compute policy loss  
        current_act = self.actor(obs_n[self.current_agent].transpose(0,1).type(torch.FloatTensor).to(self.device)) # batch_size x size
        act_n[self.current_agent] = current_act.to('cpu').transpose(0,1).detach()
        act_n = torch.vstack(act_n).to(self.device)
        #act_n = torch.tensor(np.vstack(act_n)) 
        act_n = act_n.transpose(0,1).contiguous()  # batch_size x (num_agent x size)
        actor_loss = -self.critic(tensor_obs_n.float(), act_n.float()).mean()        

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        

        # update
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

        self.train_step += 1

        #if self.train_step > 0 and self.train_step % 100 == 0:
        #    self.save_model(self.train_step)

    def save_model(self, train_step):
        num = str(train_step // 100)
        model_path = os.path.join(self.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic.state_dict(),  model_path + '/' + num + '_critic_params.pkl')


    def collect_params(self):
        self.swag_model.collect_model(self.critic)

    def sample_params(self):
        self.swag_model.sample()
