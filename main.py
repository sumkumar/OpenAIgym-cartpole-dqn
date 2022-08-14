import gym
env = gym.make('CartPole-v1')
env.reset()
for _ in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        observation = env.reset()
    env.render()
env.close()

