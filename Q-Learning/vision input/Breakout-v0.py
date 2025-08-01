import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from collections import namedtuple, deque
import torch.nn.functional as F
import cv2

# 预图像输入处理
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cropped = gray[34:194]
    resized = cv2.resize(cropped,(84,84))
    return resized /255.0

# 构建DQN网络
class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136,512),
            nn.ReLU(),
            nn.Linear(512,action_dim)
        )

    def forward(self,x):
        return self.net(x)
    
# 定义经验回放池
class ReplayBuffer:
    def __init__(self,capacity):
        # 创建一个双端队列 deque，容量为 capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # 将经验存入replay buffer
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # 将多个 list 转换为 numpy 数组
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
# 用于初始化，更新状态的堆叠桢

"""
	•	stacked_frames: 一个长度为 4 的 deque，保存最近的 4 帧图像。
	•	frame: 当前从环境中得到的新图像帧（即当前状态）。
	•	is_new_episode: 是否是一个新 episode 的开头，如果是则初始化所有帧为当前帧。
"""
def update_stack_frames(stacked_frames, frame, is_new_episode):
    # 对当前帧进行预处理,上面写的函数
    frame = preprocess_frame(frame)
    # 如果是一个新的epsode，则用当前帧复制 4 份，组成一个 deque，作为初始的 4 帧图像。
    if is_new_episode:
        stacked_frames = deque([frame]*4, maxlen=4)
    # 否则，将当前帧添加到 deque 的右侧，同时移除最左侧的帧
    else:
        stacked_frames.append(frame)
    # 一个是堆叠好的帧数组，可作为 DQN 的输入， 一个是更新后的 stacked_frames，供下一步继续使用
    return np.stack(stacked_frames, axis=0), stacked_frames

# 建立replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def optimize_model(transitions, policy_net, target_net, optimizer, gamma):
    states, actions, rewards, next_states, dones = zip(*transitions)

    state_batch = torch.tensor(np.stack(states), dtype=torch.float32)
    action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    reward_batch = torch.tensor(rewards, dtype=torch.float32)
    next_state_batch = torch.tensor(np.stack(next_states), dtype=torch.float32)
    done_batch = torch.tensor(dones, dtype=torch.float32)

    q_value = policy_net(state_batch).gather(1, action_batch).squeeze()

    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1)[0]
        target_q_value = reward_batch + gamma * next_q_values * (1 - done_batch)
    
    loss = F.smooth_l1_loss(q_value, target_q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#初始化环境
gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
action_dim = env.action_space.n

#初始化网络与buffer
#创造一个主网络（策略）
policy_net = DQN(action_dim)
#创造一个目标网络
target_net = DQN(action_dim)
# 将主网络的权重复制给目标网络
target_net.load_state_dict(policy_net.state_dict())
# 评估模式
target_net.eval()

#为主网络 policy_net 设置一个 Adam 优化器，用于训练
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

# 初始化经验回放池
replay_buffer = ReplayBuffer(capacity=10000)

#主循环结构
num_episodes = 10000 # 训练集大小
batch_size = 32 # 批次大小
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 探索速率
epsilon_decay = 0.995  # 探索速率的衰减率
min_epsilon = 0.01  # 最小探索速率
target_update = 10 # 目标网络更新频率

#主训练循环
for episode in range(num_episodes):
    # 重置环境，开始新的 episode
    state = env.reset()[0]
    # 初始化一个长度为 4 的 deque，每个元素是 84x84 的全零图像（预处理后大小）。
    stack_frames = deque([np.zeros((84,84))]*4, maxlen=4)
    # 对第一帧图像进行预处理，将该帧复制 4 份作为初始堆叠帧，因为是新 episode（第三个参数为 True）
    # 这里调用了你定义的 stack_frames() 函数，第三个参数是 True，表示是新 episode。这时候会用当前的 state （实际是第一帧 RGB 图像）进行如下操作：
    # 把处理后的图像 frame（灰度化、缩放等）复制 4 份，替换原来的全 0，生成一个新的 stacked_frames
    state, stack_frames = update_stack_frames(stack_frames, state, True)

    total_reward = 0
    done = False

    while not done:
        # 定义epsilon-greedy策略
        if random.random() < epsilon:
            # 随机选择动作
            action = env.action_space.sample()
        else:
            # 选择最优动作
            # 将 state（一个 NumPy 数组，形状是 (4, 84, 84)）转换成 PyTorch 的 Tensor。
            # 神经网络在推理时通常期望输入是 批量数据。即使你只输入一张图片，也要包装成 shape 为 (1, C, H, W) 的形式。
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = q_values.max(1)[1].item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state_stack, stacked_frames = update_stack_frames(stack_frames, next_state, False)
        replay_buffer.push(state, action, reward, next_state_stack, done)
        state = next_state_stack
        total_reward += reward

        if len(replay_buffer) >= batch_size:
            transitions = replay_buffer.sample(batch_size)
            optimize_model(transitions, policy_net, target_net, optimizer, gamma)

        if done:
            break
    if episode > min_epsilon:
        epsilon *= epsilon_decay
    
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")
