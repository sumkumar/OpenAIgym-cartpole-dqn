# https://www.gymlibrary.ml/environments/classic_control/cart_pole/
import gym
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd

import DQN

show = False
vis_window = 100    # Visualization window

env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
exp_replay_size = 10000
agent = DQN.DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, 64, output_dim], lr=1e-3, sync_freq=5,
                      exp_replay_size=exp_replay_size)

# initialize experience replay
index = 0
for i in range(exp_replay_size):
    obs = env.reset()
    if show:
        env.render()
    done = False
    while done != True:
        A = agent.get_action(obs, env.action_space.n, epsilon=1)
        obs_next, reward, done, _ = env.step(A.item())
        agent.collect_experience([obs, A.item(), reward, obs_next])
        obs = obs_next
        index += 1
        if index > exp_replay_size:
            break

# Main training loop
losses_list, reward_list, episode_len_list, epsilon_list = [], [], [], []
index = 128
episodes = 10000
epsilon = 1

for i in tqdm(range(episodes)):
    obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
    while done != True:
        ep_len += 1
        A = agent.get_action(obs, env.action_space.n, epsilon)
        obs_next, reward, done, _ = env.step(A.item())
        if show:
            env.render()
        # if done:
        #     print("\nEpisode finished after {} timesteps".format(ep_len + 1))
        agent.collect_experience([obs, A.item(), reward, obs_next])

        obs = obs_next
        rew += reward
        index += 1

        if index > 128:
            index = 0
            for j in range(4):
                loss = agent.train(batch_size=128)
                losses += loss
    if epsilon > 0.05:
        epsilon -= (1 / 5000)

    losses_list.append(losses / ep_len), reward_list.append(rew), episode_len_list.append(ep_len), epsilon_list.append(epsilon)

df = pd.DataFrame(list(zip(losses_list, reward_list, episode_len_list, epsilon_list)),
               columns =['Loss', 'Reward', 'Ep_len', 'Epsilon'])

# df['index'] = range(1, len(df) + 1)

Running_avgs = pd.DataFrame(list(zip(df['Loss'].rolling(vis_window).mean(), df['Reward'].rolling(vis_window).mean(), df['Ep_len'].rolling(vis_window).mean())), columns=['Loss_avg', 'Reward_avg', 'Ep_len_avg'])

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

sns.lineplot(data=Running_avgs)
plt.show()

