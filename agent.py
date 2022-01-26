'''
Implementation inspired by:

1. https://www.youtube.com/watch?v=H9uCYnG3LlE, last visited: 26.01.2022
2. https://github.com/rlcode/per, last visited: 26.01.2022
3. https://ieeexplore.ieee.org/document/9166560, last visited: 26.01.2022
'''

from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt


MODEL_SAVE_PATH = './trained_q_model/q_model_episode_{}.pth'

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions):
        super(DeepQNetwork, self).__init__()
        self.n_actions = n_actions
        self.conv_layer1 = nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = 10, stride = 8)
        self.conv_layer2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2)
        self.conv_layer3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)

        self.fully_connected_layer1 = nn.Linear(8192, 1, dtype = torch.float32)

        self.fully_connected_layer2 = nn.Linear(8192, self.n_actions, dtype = torch.float32)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss(reduction = 'none')
        self.device = torch.device('cuda:0')
        self.to(self.device)

    def calculate_loss(self, eval, target, td_error, isw):
        return isw * td_error * torch.mean((eval - target) ** 2).to(self.device)

    def forward(self, input):
        input = F.layer_norm(input, input.shape[1:])
        conv1_output = F.relu(self.conv_layer1(input))

        output_conv_layer2 = self.conv_layer2(conv1_output)
        conv2_output = F.relu(F.layer_norm(output_conv_layer2, output_conv_layer2.shape[1:]))

        output_conv_layer3 = self.conv_layer3(conv2_output)
        conv3_output = F.relu(F.layer_norm(output_conv_layer3, output_conv_layer3.shape[1:]))

        conv3_output = conv3_output.view(-1, 8192)

        v = self.fully_connected_layer1(conv3_output)
        advantage = self.fully_connected_layer2(conv3_output)

        q = v + advantage - torch.mean(advantage)
        return q

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_loss(self):
        return self.loss
    
    def set_loss(self, loss):
        self.loss = loss


class Agent():
    def __init__(self, gamma, epsilon, alpha, input_dims, experience_memory_size, batch_size, n_actions, eps_end = 0.01, eps_dec = 5e-4, main_network = None, target_network = None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.LOSS = deque()
        self.action_space = [i for i in range(n_actions)]
        self.experience_memory_size = experience_memory_size

        self.Q = DeepQNetwork(self.alpha, n_actions) if main_network is None else main_network
        self.Q_target = DeepQNetwork(self.alpha, n_actions) if target_network is None else target_network

    def get_next_action(self, observation, training):
        if (not training) or np.random.random() > self.epsilon:
            state = torch.tensor(np.array([observation])).to(self.Q.device)
            actions = self.Q(state)
            print('MAX: {}'.format(actions))
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
            print('RANDOM: {}'.format(action))
        
        return action

    def update_target(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def reduce_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end
        
    def optimize_q_network(self, experience_replay_buffer, episode):
        print('Learn')
        batch, sample_indices, importance_sampling_weights = experience_replay_buffer.sample()

        states = torch.tensor(np.array([batch_row[0] for batch_row in batch], dtype = np.float32)).unsqueeze(1).to(self.Q.device)
        next_states = torch.tensor(np.array([batch_row[1] for batch_row in batch], dtype = np.float32)).unsqueeze(1).to(self.Q.device)
        actions = torch.tensor(np.array([batch_row[2] for batch_row in batch]), dtype = torch.long).to(self.Q.device)
        rewards = torch.tensor(np.array([batch_row[3] for batch_row in batch], dtype = np.float32)).to(self.Q.device).unsqueeze(1)
        dones = torch.tensor(np.array([batch_row[4] for batch_row in batch], dtype = np.bool8)).to(self.Q.device).unsqueeze(1)
                    
        #print('States')
        #print(states)
        #print('Actions')
        #print(actions)
        #print('Rewards')
        #print(rewards)
        #print(next_states)
        #print(sample_indices)

        q_eval = torch.zeros((self.batch_size, self.n_actions)).to(self.Q.device)
        q_next = torch.zeros((self.batch_size, self.n_actions)).to(self.Q_target.device)
        q_next_states = torch.zeros((self.batch_size, self.n_actions)).to(self.Q.device)

        for index in range(self.batch_size):
            q_eval[index] = self.Q(states[index])
            q_next[index] = self.Q_target(next_states[index])
            q_next_states[index] = self.Q(next_states[index])
        
        q_eval = q_eval.gather(-1, actions.view(-1,1)).to(self.Q.device)
        max_action_indices = torch.argmax(q_next_states, dim = 1)
        q_next = q_next.gather(-1, max_action_indices.view(-1,1)).to(self.Q_target.device)

        q_target = rewards + (self.gamma * q_next * (1 - dones.long()))
        
        td_errors = q_target - q_eval

        self.Q.optimizer.zero_grad()

        for index in np.arange(sample_indices.size):
            elem_index = sample_indices[index]
            td_error = td_errors[index].item()
            priority = abs(td_error) + 0.01
            experience_replay_buffer.update_priority_at(elem_index, priority)

        loss_output = (torch.FloatTensor(importance_sampling_weights).to(self.Q.device) * self.Q.loss(q_eval, q_target).to(self.Q.device)).mean()
        loss_output.backward()

        self.Q.optimizer.step()

    def save_model(self, episode):
        file_name = MODEL_SAVE_PATH
        file_name = file_name.format(episode)

        torch.save(
            {
            'epoch': episode,
            'main_state_dict': self.Q.state_dict(),
            'target_state_dict': self.Q_target.state_dict(),
            'main_optimizer_state_dict': self.Q.get_optimizer().state_dict(),
            'target_optimizer_state_dict': self.Q_target.get_optimizer().state_dict(),
            'main_loss': self.Q.get_loss(),
            'target_loss': self.Q_target.get_loss()
            }, file_name
        )

