import random
from collections import deque

class ExperienceReplayBuffer:
    def __init__(self, max_buffer_size, batch_size):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen = max_buffer_size)

    def is_possible_to_take_samples(self):
        return len(self.replay_buffer) >= self.batch_size
    
    def add_experience(self, experience):
        self.replay_buffer.append(experience)

    def sample(self):
        return random.sample(self.replay_buffer, self.batch_size)

