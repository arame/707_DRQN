import numpy as np
from config import Config

class ReplayBuffer():
    def __init__(self, input_shape, n_actions):
        self.mem_cntr = 0
        self.state_memory = np.zeros((Config.max_mem, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((Config.max_mem, *input_shape), dtype=np.float32)

        self.action_memory = np.zeros(Config.max_mem, dtype=np.int64)
        self.reward_memory = np.zeros(Config.max_mem, dtype=np.float32)
        self.terminal_memory = np.zeros(Config.max_mem, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_new, done):
        index = self.mem_cntr % Config.max_mem
        self.state_memory[index] = state
        self.new_state_memory[index] = state_new
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, Config.max_mem)
        batch = np.random.choice(max_mem, Config.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
