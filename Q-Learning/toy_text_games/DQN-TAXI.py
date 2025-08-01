import gymnasium as gym
import torch.nn as nn
from collections import namedtuple, deque
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 创建环境
env = gym.make("Taxi-v3")
state, info = env.reset()

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(state_size, 64)  # 将状态编号映射到向量
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.embedding(state)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 建立replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# 写 ε-greedy 策略
class EpsilonGreedyStrategy:
    def __init__(self, eps_start=1.0, eps_end=0.05, decay=0.01):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay = decay
    
    def get_epsilon(self, current_step):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.decay * current_step)

# 选择动作
def select_action(state, policy_net, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        with torch.no_grad():
            return policy_net(torch.tensor([state])).max(1)[1].item()

# 优化模型
def optimize_model(transitions, policy_net, target_net, optimizer, gamma):
    batch = Transition(*zip(*transitions))

    state_batch = torch.tensor(batch.state)
    action_batch = torch.tensor(batch.action).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward)
    next_state_batch = torch.tensor(batch.next_state)
    done_batch = torch.tensor(batch.done, dtype=torch.float32)

    q_value = policy_net(state_batch).gather(1, action_batch).squeeze()

    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

    loss = F.smooth_l1_loss(q_value, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 初始化各种组件
state_size = env.observation_space.n
action_size = env.action_space.n

policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())  # 初始化 target_net
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
replay_buffer = ReplayBuffer(capacity=10000)
strategy = EpsilonGreedyStrategy()

# 主循环参数
num_episodes = 10000
batch_size = 128
gamma = 0.99  # 折扣因子
target_update_freq = 10  # 每10个episode更新一次target_net

episode_rewards = []  # 记录每个 episode 的 total reward

# 主训练循环
for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0

    for t in range(200):  # 限制每个ep的最大长度
        epsilon = strategy.get_epsilon(episode)
        action = select_action(state, policy_net, epsilon, action_size)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(replay_buffer) >= batch_size:
            transitions = replay_buffer.sample(batch_size)
            optimize_model(transitions, policy_net, target_net, optimizer, gamma)

        if done:
            break

    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    episode_rewards.append(total_reward)
    print(f"Episode {episode}, Total Reward: {total_reward}")

# 绘制训练曲线
def plot_rewards(rewards, window=10):
    rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10,5))
    plt.plot(rolling_avg)
    plt.title(f"Rolling Average Reward (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()

plot_rewards(episode_rewards)