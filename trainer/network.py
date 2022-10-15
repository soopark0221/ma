import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )        
    
    def forward(self, obs):
        x = self.actor(obs)
        x = th.tanh(x)
        return x

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
			nn.Linear(512, 128),
            nn.ReLU(),
        )        
        self.last = nn.Linear(128, 1)
    
    def forward(self, obs, act):
        x = self.critic(th.cat([obs, act], dim=-1))
        x = self.last(x)
        return x


class Critic_MA(nn.Module):
	def __init__(self, n_agent, dim_observation, dim_action, norm_in=True, hidden=64):
		super(Critic_MA,self).__init__()
		# normalize inputs
		self.dim_observation = dim_observation
		self.dim_action = dim_action
		obs_dim = self.dim_observation * n_agent
		act_dim = self.dim_action * n_agent
		
		if norm_in:
			self.in_fn = nn.BatchNorm1d(obs_dim+act_dim)
			self.in_fn.weight.data.fill_(1)
			self.in_fn.bias.data.fill_(0)
		else:
			self.in_fn = x


		self.FC1 = nn.Linear(obs_dim+act_dim,hidden)
		self.FC2 = nn.Linear(hidden,hidden)
		self.FC3 = nn.Linear(hidden,1)
		
	# obs:batch_size * obs_dim
	def forward(self, obs, acts):
		result = self.in_fn(th.cat([obs, acts], dim=1))
		result = F.relu(self.FC1(result))
		result = F.relu(self.FC2(result))
		result = self.FC3(result)
		return result
		
class Actor_MA(nn.Module):
	def __init__(self, dim_observation, dim_action, norm_in=True, hidden=64):
		super(Actor_MA,self).__init__()

		#print('model.dim_action',dim_action) 
		if norm_in:  # normalize inputs
			self.in_fn = nn.BatchNorm1d(dim_observation)
			self.in_fn.weight.data.fill_(1)
			self.in_fn.bias.data.fill_(0)
		else:
			self.in_fn = lambda x: x

		self.FC1 = nn.Linear(dim_observation, hidden)
		self.FC2 = nn.Linear(hidden, hidden)
		self.FC3 = nn.Linear(hidden, dim_action)
		

	def forward(self, obs):
		result = self.in_fn(obs)
		result = F.relu(self.FC1(result))
		result = F.relu(self.FC2(result))
		#result = F.tanh(self.FC3(result))
		result = self.FC3(result)
		return result

class EnsembleCrt_MA(nn.Module):
	def __init__(self, n_agent, obs_dim, action_dim, n_ensemble=5):
		super().__init__()
		self.n_agent=n_agent
		self.obs_dim=obs_dim
		self.act_dim=action_dim
		self.n_ensemble=n_ensemble

		self.ensemble=nn.ModuleList([Critic_MA(n_agent,obs_dim,action_dim) for _ in range(n_ensemble)])

	def forward(self, obs,act, k=None):
		if k is None:
			return [net(obs,act) for net in self.ensemble]
		else:
			return self.ensemble[k](obs,act)
	
	@property
	def ensemble_size(self):
		return self.n_ensemble
