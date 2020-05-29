import argparse, os
import gym
import numpy as np
import agents as Agents
from utils import plot_learning_curve, make_env
from config import Config

#def main():
print("** STARTED **")
Config.create_directories()
env = make_env(env_name=Config.env_name, repeat=Config.repeat, clip_rewards=Config.clip_rewards,
                no_ops=Config.no_ops, fire_first=Config.fire_first)
best_score = -np.inf
agent_ = getattr(Agents, Config.algo)
agent = agent_(input_dims=env.observation_space.shape, n_actions=env.action_space.n)

if Config.load_checkpoint:
    agent.load_models()

#scores_file = fname + '_scores.npy'

scores, eps_history = [], []
n_steps = 0
steps_array = []
for i in range(Config.n_games):
    done = False
    observation = env.reset()
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_new, reward, done, info = env.step(action)
        score += reward

        if not Config.load_checkpoint:
            agent.store_transition(observation, action, reward, observation_new, int(done))
            agent.learn()
        observation = observation_new
        n_steps += 1
    scores.append(score)
    steps_array.append(n_steps)

    avg_score = np.mean(scores[-100:])
    print('episode: ', i,'score: ', score, ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
        'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

    if avg_score > best_score:
        if not Config.load_checkpoint:
            agent.save_models()
        best_score = avg_score

    eps_history.append(agent.epsilon)
    if Config.load_checkpoint and n_steps >= 18000:
        break

x = [i+1 for i in range(len(scores))]
plot_learning_curve(steps_array, scores, eps_history, Config.figure_file)
    #np.save(scores_file, np.array(scores))

#    if __name__ == '__main__':
#       main()
