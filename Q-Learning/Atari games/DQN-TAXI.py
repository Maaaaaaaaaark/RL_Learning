import gymnasium as gym
import torch.nn as nn
from collections import namedtuple
import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F


env = gym.make("Taxi-v3", render_mode="human")
state, info = env.reset()


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
#定义一个Transition 元组，表示一次 agent 的经验
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        #固定大小的replay buffer
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # 将经验存入replay buffer
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 从replay buffer中随机抽取经验
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        # 返回replay buffer中的经验数量
        return len(self.buffer)

# 写 ε-greedy 策略
        """
        初始化ε-greedy策略
        :param eps_start: 初始epsilon（完全随机）
        :param eps_end: 最小epsilon（训练后期接近确定性策略）
        :param decay: 衰减系数（越大下降越快）
        """
class EpsilonGreedyStrategy:
    def __init__(self, eps_start=1.0, eps_end=0.05, decay=0.01):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay = decay
    
    def get_epsilon(self, current_step):
        # 指数衰减函数，就是前中后期都采用不用的Ep，保证探索性
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.decay * current_step)

def select_action(state, policy_net, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        with torch.no_grad():
            return policy_net(torch.tensor([state])).max(1)[1].item()


#用于从 replay buffer 中采样一个 mini-batch，并根据 DQN loss 执行一次梯度下降更新
def optimize_model(transitions, policy_net, target_net, optimizer, gamma):
    batch = Transition(*zip(*transitions))

    # 将batch中的每一项都转换成tensor
    state_batch = torch.tensor(batch.state)
    action_batch = torch.tensor(batch.action).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward)
    next_state_batch = torch.tensor(batch.next_state)
    done_batch = torch.tensor(batch.done, dtype=torch.float32)

    # 计算当前的Q值 policy_net(state).gather(1, action) 表示 Q(s,a)
    q_value = policy_net(state_batch).gather(1, action_batch).squeeze()

    # 目标的Q值 r + γ * max_a' Q_target(s', a')，如果 done=1 则不加未来奖励
    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

    # 计算loss
    loss = F.smooth_l1_loss(q_value, target_q_values)

    # 梯度下降
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#初始化各种组件
state_size = env.observation_space.n
action_size = env.action_space.n

policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())  # 初始化 target_net
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
replay_buffer = ReplayBuffer(capacity=10000)
strategy = EpsilonGreedyStrategy()

# 主循环
num_episodes = 500
batch_size = 64
gamma = 0.99  # 折扣因子
target_update_freq = 10  # 每10个episode更新一次target_net

for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0

    for t in range (200): #限制每个ep的最大长度
        epsilon = strategy.get_epsilon(episode)
        # 选择动作
        action = select_action(state, policy_net, episode, action_size)

        # 执行动作，获取下一个状态，奖励和是否终止标志
        next_state, reward, terminated, truncated, info = env.step(action)

        # 存储经验
        done = terminated or truncated

        # 将经验存入replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        # 更新状态
        state = next_state
        total_reward += reward

        # d当buffer足够大的时候才开始训练
        if len(replay_buffer) >= batch_size:
            transitions = replay_buffer.sample(batch_size)
            #训练模型
            optimize_model(transitions, policy_net, target_net, optimizer, gamma)

        if done:
            break

    # 更新target net
    if episode % target_update_freq == 0:
        # policy_net.state_dict()获取policy的网络策略给target_net
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}")





