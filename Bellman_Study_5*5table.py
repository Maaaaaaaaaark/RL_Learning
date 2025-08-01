import gymnasium as gym

# Create our training environment - a cart with a pole that needs balancing
# 创建环境，指定环境的可视化放松
env = gym.make('CartPole-v1',render_mode="human")

# Reset environment to start a new episode
# 重制环境获取第一个观察值以及其他的信息，就像是开始一场新的游戏或新一集一样，要使用随机种子或者选项初始化环境
observation, info = env.reset()

print(f"开始观察: {observation}")

# 用episode_over去卡while 循环
episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = push cart left, 1 = push cart right
    action = env.action_space.sample() # 随机选择动作

    #take an action and see what happened
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)
    total_reward += reward
    #在terminated（到达一定的步长）或者truncated（回合结束）为true的情况下
    episode_over = terminated or truncated

print(f"专辑完成！总奖励！: {total_reward}")
env.close()