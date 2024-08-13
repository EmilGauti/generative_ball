import torch
import torch.nn as nn
import numpy as np

class network(nn.Module):
    def __init__(self,state_size,action_size):
        super(network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_actions = action_size

        self.fc=nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )

    def forward(self, x):
        return self.fc(x)

class DQN:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.n_actions = action_size
        self.lr = 1e-7
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.001
        self.batch_size = 32
        self.verbose = False
        
        
        # We define our memory buffer where we will store our experiences
        # We stores only the 2000 last time steps
        self.memory_buffer= list()
        self.max_memory_buffer = 2000
        
        # We creaate our model having to hidden layers of 24 units (neurones)
        # The first layer has the same size as a state size
        # The last layer has the size of actions space

        self.model = network(self.state_size, self.action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def compute_action(self, current_state):
        # We sample a variable uniformly over [0,1]
        # if the variable is less than the exploration probability
        #     we choose an action randomly
        # else
        #     we forward the state through the DNN and choose the action 
        #     with the highest Q-value.
        if np.random.uniform(0,1) < self.exploration_proba:

            return np.random.choice(range(self.n_actions))
        if torch.is_tensor(current_state) == False:
            current_state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(current_state)[0]

        return torch.argmax(q_values).item()
    
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
        if self.verbose:
            print("exploration_prob",self.exploration_proba)
    
    # At each time step, we store the corresponding experience
    def store_episode(self,current_state, action, reward, next_state, done):
        #We use a dictionnary to store them
        self.memory_buffer.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done" :done
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)
    def train_step(self, current_state, q_current_state, verbose=False):
        if torch.is_tensor(current_state) == False: 
            current_state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
        self.optimizer.zero_grad()
        loss = self.criterion(q_current_state, self.model(current_state))
        loss.backward()
        self.optimizer.step()
        
        return loss
    def train(self):
        # We shuffle the memory buffer and select a batch size of experiences
        #np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]
        
        loss=0
        for experience in batch_sample:
            # We compute the Q-values of S_t
            tmp = experience["current_state"]
            if torch.is_tensor(tmp) == False:
                tmp=torch.tensor(tmp, dtype=torch.float32).unsqueeze(0)
            q_current_state = self.model(tmp)
            
            # We compute the Q-target using Bellman optimality equation
            q_target = experience["reward"]
            if not experience["done"]:
                tmp=torch.tensor(experience["next_state"], dtype=torch.float32).unsqueeze(0)
                q_target = q_target + self.gamma*torch.max(self.model(tmp)[0])


            q_current_state[0][0][experience["action"]] = q_target

            loss+=self.train_step(experience["current_state"], q_current_state, verbose=self.verbose)
        return loss