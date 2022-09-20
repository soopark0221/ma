import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )        
        self.last = nn.Linear(hidden_size, 1)
    
    def forward(self, obs, act):
        x = self.critic(th.cat([obs, act], dim=-1))
        x = self.last(x)
        return x


class Critic_MA(nn.Module):
	def __init__(self,n_agent,dim_observation,dim_action):
		super(Critic_MA,self).__init__()
		self.n_agent = n_agent
		self.dim_observation = dim_observation
		self.dim_action = dim_action
		obs_dim = self.dim_observation * n_agent
		act_dim = self.dim_action * n_agent
		
		self.FC1 = nn.Linear(obs_dim,1024)
		self.FC2 = nn.Linear(1024+act_dim,512)
		self.FC3 = nn.Linear(512,300)
		self.FC4 = nn.Linear(300,1)
		
	# obs:batch_size * obs_dim
	def forward(self, obs, acts):
		result = F.relu(self.FC1(obs))
		combined = th.cat([result, acts], dim=1)
		result = F.relu(self.FC2(combined))
		return self.FC4(F.relu(self.FC3(result)))
		
class Actor_MA(nn.Module):
	def __init__(self,dim_observation,dim_action):
		#print('model.dim_action',dim_action) 
		super(Actor_MA,self).__init__()
		self.FC1 = nn.Linear(dim_observation,500)
		self.FC2 = nn.Linear(500,128)
		self.FC3 = nn.Linear(128,dim_action)
		

	def forward(self,obs):
		result = F.relu(self.FC1(obs))
		result = F.relu(self.FC2(result))
		result = F.tanh(self.FC3(result))
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
