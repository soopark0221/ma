from collections import namedtuple
import random
import numpy as np

Experience = namedtuple('Experience',
						('states','actions','next_states','rewards'))
BootExp=namedtuple('BootExp',
					('states','actions','next_states','rewards','masks'))

class ReplayMemory:
	def __init__(self,capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		
	def push(self,*args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)

		self.memory[self.position] = Experience(*args)
		self.position = int((self.position + 1)%self.capacity)
		
	def sample(self,batch_size):
		# print(len(self.memory),batch_size)
		return random.sample(self.memory,batch_size)
	
	def __len__(self):
		return len(self.memory)

class BootMemory:
	def __init__(self, capacity, n_ensemble=5, p=0.9):
		self.n_ensemble=n_ensemble
		self.bern_p=p
		self.capacity = capacity
		self.memory = []
		self.position = 0
		
		if n_ensemble==1:
			self.bern_p=1.0
		
	def push(self,*args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		state=args[0]
		action=args[1]
		next_s=args[2]
		reward=args[3]
		mask=np.random.binomial(1,self.bern_p,self.n_ensemble)
		exp=(state,action,next_s,reward,mask)
		self.memory[self.position] = BootExp(*exp)
		self.position = int((self.position + 1)%self.capacity)
		
	def sample(self,batch_size):
		# print(len(self.memory),batch_size)
		return random.sample(self.memory,batch_size)
	
	def __len__(self):
		return len(self.memory)