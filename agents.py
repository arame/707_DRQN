import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer
from config import Config

class Agent():
    def __init__(self, input_dims, n_actions):
        self.epsilon = Config.epsilon
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(input_dims, n_actions)

    def store_transition(self, state, action, reward, state_new, done):
        self.memory.store_transition(state, action, reward, state_new, done)

    def choose_action(self, observation):
        raise NotImplementedError

    def replace_target_network(self):
        if self.learn_step_counter % Config.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * Config.eps_decay, Config.eps_min)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer()

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def learn(self):
        raise NotImplementedError

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DeepQNetwork(self.n_actions, input_dims=self.input_dims, name=Config.get_name('_q_eval'))
        self.q_next = DeepQNetwork(self.n_actions, input_dims=self.input_dims, name=Config.get_name('_q_next'))

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < Config.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(Config.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + Config.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decay_epsilon()

