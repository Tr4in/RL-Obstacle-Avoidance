import unreal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, layer1_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer1_dims = layer1_dims
        self.n_actions = n_actions
        self.layer1 = nn.Linear(self.input_dims, self.n_actions, dtype = torch.float32)
        self.optimizer = optim.SGD(self.parameters(), lr = lr)
        self.loss = nn.MSELoss(reduction = 'sum')
        self.device = torch.device('cuda:0')
        self.to(self.device)

    def forward(self, input):
        actions = F.relu(self.layer1(input))
        return actions

class Agent():
    def __init__(self, unreal_agent, agent_speed, gamma, epsilon, alpha, input_dims, experience_memory_size, batch_size, n_actions, 
        max_mem_size = 1000, eps_end = 0.01, eps_dec = 5e-4):
        self.gamma = gamma
        self.unreal_agent = unreal_agent
        self.speed = agent_speed
        self.epsilon = epsilon
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.experience_replay_counter = 0
        self.input_dims = input_dims
        self.experience_memory_size = experience_memory_size
        

        self.Q = DeepQNetwork(self.alpha, input_dims, n_actions, n_actions)
        #self.state_memory = np.zeros((self.mem_size, input_dims), dtype = np.double)
        self.experience_memory_state = np.zeros((self.experience_memory_size, input_dims), dtype = np.float32) # make float
        self.experience_memory_next_state = np.zeros((self.experience_memory_size, input_dims), dtype = np.float32) # make float
        self.experience_memory_action = np.zeros(self.experience_memory_size, dtype = np.int32)
        self.experience_memory_reward = np.zeros(self.experience_memory_size, dtype = np.float32)
        self.network_target_parameters = self.Q.parameters() # TODO Paremeters are not set for Fixed Q target
        self.Q_target = DeepQNetwork(self.alpha, input_dims, n_actions, n_actions)


    def store_transition(self, state, action, reward, next_state):
        index = self.experience_replay_counter
        np.put(self.experience_memory_state[index], range(self.input_dims), state) 
        self.experience_memory_next_state[index] = next_state
        self.experience_memory_reward[index] = reward
        self.experience_memory_action[index] = action

        self.experience_replay_counter += 1


    def compute_learning_target(self):
        if self.batch_size > self.experience_memory_size:
            self.batch_size = self.experience_memory_size
        
        self.Q_target.optimizer.zero_grad()
        sample_indices = random.sample(range(self.experience_memory_size), self.batch_size)

        for sample_index in sample_indices:
            state = torch.tensor(self.experience_memory_state[sample_index]).to(self.Q_target.device)
            action = torch.tensor(self.experience_memory_action[sample_index]).to(self.Q_target.device)
            next_state = torch.tensor(self.experience_memory_next_state[sample_index]).to(self.Q_target.device)
            reward = torch.tensor(self.experience_memory_reward[sample_index]).to(self.Q_target.device)

            q_eval = self.Q_target.forward(state)[action]
            q_next = self.Q_target.forward(next_state)

            q_target = reward + self.gamma * torch.max(q_next).item()

            loss_output = self.Q_target.loss(q_target, q_eval).to(self.Q_target.device)
            loss_output.backward()
            self.Q_target.optimizer.step()


    def get_next_action(self, observation):
        if np.random.random() > self.epsilon:
             state = torch.tensor([observation]).to(self.Q_target.device)
             actions = self.Q_target.forward(state)
             action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        if action == 1:
            unreal.log("ACTION: LEFT")
            self.unreal_agent.set_actor_rotation(unreal.Rotator(0, 0, -45.0), True)
        
        if action == 2:
            unreal.log("ACTION: RIGHT")
            self.unreal_agent.set_actor_rotation(unreal.Rotator(0, 0, 45.0), True)

        self.unreal_agent.set_actor_location(self.unreal_agent.get_actor_location() + self.unreal_agent.get_actor_forward_vector() * self.speed, True, False)

        return action

    def swap_models(self):
        Q_target_copy = self.Q_target
        self.Q_target = self.Q
        self.Q = Q_target_copy

    def optimize_q_network(self):
        self.Q.optimizer.zero_grad()
        sample_indices = random.sample(range(self.experience_memory_size), self.batch_size)

        for sample_index in sample_indices:
            state = torch.tensor(self.experience_memory_state[sample_index]).to(self.Q.device)
            action = torch.tensor(self.experience_memory_action[sample_index]).to(self.Q.device)
            next_state = torch.tensor(self.experience_memory_next_state[sample_index]).to(self.Q.device)
            reward = torch.tensor(self.experience_memory_reward[sample_index]).to(self.Q.device)

            q_eval = self.Q.forward(state)[action]
            q_next = self.Q_target.forward(next_state)

            q_target = reward + self.gamma * torch.max(q_next).item()

            loss_output = self.Q.loss(q_target, q_eval).to(self.Q_target.device)
            loss_output.backward()
            self.Q.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end