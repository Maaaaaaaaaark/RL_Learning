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
    # 原始RGB转灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 裁剪游戏区域
    cropped = gray[34:194,:]
    # 缩放到84*84
    resized = cv2.resize(cropped,(84,84))
    #转float 并且归一化
    # 是将像素类型从整数转为浮点数，为了方便送入神经网络
    normalized = resized.astype(np.float32) / 255.0
    return normalized

# 定义神经网络：
class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.actor = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):  # x.shape = (B, 1, 84, 84)
        features = self.shared_conv(x)
        features = self.flatten(features)
        return self.actor(features), self.critic(features)

#初始化环境
gym.register_envs(ale_py)
env = gym.make('ALE/MarioBros-v5', render_mode='human')
action_dim = env.action_space.n # 动作纬度
state_dim = env.observation_space.shape[0] # 状态纬度

"""
	•	self.states: 存储所有时间步的状态 s_t
	•	self.actions: 存储所有时间步采取的动作 a_t
	•	self.log_probs: 存储所有时间步动作对应的对数概率，用于 PPO 的 ratio
	•	self.rewards: 存储所有时间步获得的奖励 r_t
	•	self.values: 存储 Critic 网络估算的状态价值 V(s_t)，用于计算优势值（Advantage）
	•	self.dones: 存储是否为 episode 结束（1 表示结束，0 表示未结束）

"""
# buffer
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def clear(self):
        print(f"Clearing buffer with {len(self.states)} steps")
        self.__init__()

# Agent 
"""
            state_dim: 状态空间
            action_dim：动作空间
            hidden_dim：隐藏层
            lr_actor：Actor网络的学习率
            lr_critic：Critic网络的学习率
            gamma： 折扣因子
            lam： GAE 参数（计算优势函数的因子）
            eps_clip ： PPO裁剪参数（限制策略的更新）
"""

class PPOAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim,
            lr_actor = 3e-4,
            lr_critic = 1e-3,
            gamma = 0.99,
            lam = 0.95,
            eps_clip = 0.2
                 ):
        # 送入神经网络当中获取策略
        self.policy = ActorCritic(action_dim)
        # PPO 算法中的 Actor 和 Critic 网络分别设置优化器参数，以便后续进行反向传播和梯度更新
        self.optimizer = torch.optim.Adam([
            {'params':self.policy.actor.parameters(),'lr':lr_actor},
            {'params':self.policy.critic.parameters(),'lr':lr_critic}
        ])
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.buffer = RolloutBuffer()
    
    def select_action(self, state):
        #从神经网络中选择做的动作
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        """
    •	probs：由 actor 网络生成的每个动作的概率
	•	value：由 critic 网络估算的当前状态的价值
        """
        with torch.no_grad():
            probs, value = self.policy(state)
        #构造一个基于 probs 的离散分布 dist
        dist = torch.distributions.Categorical(probs)
        #从动作概率中选择动作
        action = dist.sample()
        #计算动作对应的对数概率，为后面对吧服务
        log_prob = dist.log_prob(action)

        # 将状态动作对数概率 和状态价值存入 buffer
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value)

        return action.item()
    # 计算优势函数
    def computer_gae(self, next_value):
        values = self.buffer.values + [next_value]
        gae = 0 
        advantages = []
        # 从最后一步往前遍历所有时间步，因为 GAE 是一个递归公式，从后往前累加
        for t in reversed(range(len(self.buffer.rewards))):
            """
            	•	values[t] 是当前状态的 V 值；
	            •	values[t+1] 是下一状态的 V 值；
	            •	(1 - self.buffer.dones[t]) 是防止终止状态继续累积未来值（如果是终止状态，那未来值为 0）；
	            •	self.gamma 是折扣因子。
            """
            delta = self.buffer.rewards[t]+self.gamma*values[t+1]*(1-self.buffer.dones[t])-values[t]
            gae = delta + self.gamma*self.lam*(1-self.buffer.dones[t])* gae
            # 把当前 timestep 的 advantage 插入列表开头；因为我们是从后往前遍历，所以插入头部可以保持时间顺序一致
            advantages.insert(0, gae)
        return advantages
    
    # 更新策略
    def update(self):
        #向前传播
        with torch.no_grad():
            #取self.buffer.states这个列表中最后一个状态的，也就是最新的状态
            next_value = self.buffer.states[-1]
            # 调用神经网络对next_state进行一次向前传播，但是由于self.policy(next_state)返回一个(action_probs, state_value)，我们只需要state_value
            _, next_value = self.policy(next_value)

        #计算优势函数
        advantage = self.computer_gae(next_value)
        # advantage是一个列表，表示每个时间步的 优势值 A(s, a）, zip(advantage, self.buffer.values) 把两个列表配对起来
        # 构造每一步的目标回报 R_t ≈ A_t + V(s_t)，用于更新 Critic 网络
        returns = [adv + val for adv, val in zip (advantage, self.buffer.values)] 

        states = torch.cat(self.buffer.states, dim=0)
        actions = torch.tensor(self.buffer.actions)
        log_probs_old = torch.stack(self.buffer.log_probs)
        returns = torch.tensor(returns).detach()
        advantage = torch.tensor(advantage).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) #normalize

        for _ in range(5):
            probs, values = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)

            rations = torch.exp(log_probs - log_probs_old.detach())
            # PPO目标函数的第一项 L^{\text{PG}} = \mathbb{E} \left[ \text{ratio} \cdot \text{Advantage} \right]
            surr1 = rations * advantage
            # PPO裁剪目标函数的第二项 L^{\text{CLIP}} = \mathbb{E} \left[ \min \left( \text{ratio} \cdot A, \text{clipped\_ratio} \cdot A \right) \right]
            surr2 = torch.clamp(rations, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # 取两个 surrogate 中较小的那个（保守策略）
            # 加负号是因为我们要最大化 PPO目标
            actor_loss = -torch.min(surr1, surr2).mean()
            # 计算 Critic 网络的损失

            # 计算 Critic 网络的均方误差损失（MSE Loss）
            critic_loss = nn.MSELoss()(values.squeeze(-1), returns)
            # 把 Actor 和 Critic 的损失合成一个总损失函数
            loss = actor_loss + 0.5 * critic_loss

            # 清除之前积累的梯度
            self.optimizer.zero_grad()
            # 执行反向传播，自动计算参数的梯度
            loss.backward()
            # 根据计算好的梯度，更新 Actor 和 Critic 网络的参数
            self.optimizer.step()
        
        self.buffer.clear()

## 运行
from tqdm import tqdm

#参数设置
hidden_dim = 128
n_episodes = 10000
max_steps = 1000  # 每回合最多多少步
update_timestep = 2000  # 每累计这么多步，更新一次
render = False          # 是否显示动画

# 初始化 PPO Agent
agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=hidden_dim
)

timestep = 0
episode_rewards = []


for episode in tqdm (range(n_episodes)):
    state, _ = env.reset()
    state = preprocess_frame(state)
    episode_reward = 0

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 存储奖励和终止标志
        agent.buffer.rewards.append(reward)
        agent.buffer.dones.append(done)
        episode_reward += reward

        next_state = preprocess_frame(next_state)

        state = next_state
        
        timestep += 1

        if timestep % update_timestep == 0:
            agent.update()

        if done:
            break
    
    episode_rewards.append(episode_reward)

    # 打印每10轮平均回报
    if (episode + 1) % 10 == 0:
        avg_reward = sum(episode_rewards[-10:]) / 10
        print(f"Episode {episode+1}, Average Reward: {avg_reward:.2f}")

env.close()




