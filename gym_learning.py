import gymnasium as gym

#创造一个动作环境
env = gym.make('CartPole-v1')
print(f"动作空间: {env.action_space}") 
print(f"观察空间: {env.observation_space}")

print(f"样本空间:{env.env.action_space.sample()}")
print(f"Sample observation: {env.observation_space.sample()}")