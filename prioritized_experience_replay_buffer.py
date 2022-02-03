'''
This class is inspired by some ideas from:
Prioritized Experience Replay: https://github.com/rlcode/per, last visited: 26.01.2022
'''

import numpy as np
from collections import deque

class PrioritizedExperienceReplayBuffer:
    def __init__(self, max_buffer_size, batch_size, alpha, beta, beta_increment):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen = max_buffer_size)
        self.priority_buffer = deque(maxlen = max_buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

    def add_experience(self, experience):
        self.priority_buffer.append(np.max(self.priority_buffer, initial=1))
        self.replay_buffer.append(experience)

    def is_possible_to_take_samples(self):
        return len(self.replay_buffer) >= self.batch_size
   
    def get_distribution_at(self, index):
        sum_priorities = np.sum(np.array(self.priority_buffer) ** self.alpha)
        return (self.priority_buffer[index] ** self.alpha / sum_priorities)

    def update_priority_at(self, index, new_priority):
        self.priority_buffer[index] = new_priority
        
    def sample(self):
        sum_priorities = np.sum(np.array(self.priority_buffer) ** self.alpha)
        distribution = (np.array(self.priority_buffer) ** self.alpha) / sum_priorities
        
        sample_indices = np.random.choice(a = np.arange(np.size(self.priority_buffer)), size = self.batch_size, p = distribution, replace = False)

        self.beta = np.min([self.beta + self.beta_increment, 1.0])

        importance_sampling_weights = (self.get_buffer_size() * distribution[sample_indices]) ** (-self.beta)
        max_weight = np.max(importance_sampling_weights)
        importance_sampling_weights = importance_sampling_weights / max_weight
        
        return np.array(self.replay_buffer)[sample_indices], sample_indices, importance_sampling_weights

