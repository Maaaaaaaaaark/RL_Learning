import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

# 定义网络
class ActorCritic(nn.Module):
    def __init__(self,state_dim,action_dim, hidden_dim):
        super(ActorCritic,self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    def forward(self,x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value
    
# 环境初始化
env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0] # 状态维度
action_dim = env.action_space.n # 动作维度

#定义存储一批轨迹（buffer）
"""
	•	self.states: 存储所有时间步的状态 s_t
	•	self.actions: 存储所有时间步采取的动作 a_t
	•	self.log_probs: 存储所有时间步动作对应的对数概率，用于 PPO 的 ratio
	•	self.rewards: 存储所有时间步获得的奖励 r_t
	•	self.values: 存储 Critic 网络估算的状态价值 V(s_t)，用于计算优势值（Advantage）
	•	self.dones: 存储是否为 episode 结束（1 表示结束，0 表示未结束）
"""
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    # 清空 buffer 中所有轨迹数据，为下一个 rollout 做准备
    def clear(self):
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
        #送入神经网络当中获取策略
        self.policy = ActorCritic(state_dim,action_dim, hidden_dim)
        #为 PPO 算法中的 Actor 和 Critic 网络分别设置优化器参数，以便后续进行反向传播和梯度更新
        #self.policy.actor.parameters()：提取 actor 网络中所有可训练参数。actor 负责输出策略（即动作概率）
        #self.policy.critic.parameters()：提取 critic 网络中所有可训练参数。critic 负责估算状态价值
        self.optimizer = torch.optim.Adam([
            {'params':self.policy.actor.parameters(),'lr':lr_actor},
            {'params':self.policy.critic.parameters(),'lr':lr_critic}
        ])
        #折扣因子
        self.gamma = gamma 
        # GAE 参数
        self.lam = lam
        # PPO裁剪参数
        self.eps_clip = eps_clip
        self.buffer = RolloutBuffer()
    
    def select_action(self, state):
        # 从神经网络中选择动作，dtype=torch.float32是因为神经网络中强行要求输入的是float32的数据
        state = torch.tensor(state,dtype=torch.float32)
        # 向前传播，进行推理，得到动作概率和状态价值（）
        with torch.no_grad():
            probs, value = self.policy(state)
        # 从神经网络获取动作概率
        dist = torch.distributions.Categorical(probs)
        # 从动作概率中选择动作
        action = dist.sample()
        # 计算动作对应的对数概率（PPO 的核心是：比较新旧策略的比值 r = π_new / π_old，这个比就是通过对数计算的）
        log_prob = dist.log_prob(action)
        
        # 将状态，动作，对数概率，状态价值存入 buffer
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value)

        return action.item()

#计算优势函数
    def computer_gae(self, next_value):
        # self.buffer.values 是一个列表，包含当前 trajectory 中每个时间步的状态价值 V(s_t)
        # next_value 是最后一个状态的 下一个状态价值，用于 bootstrap（引导）最后一步
        values = self.buffer.values + [next_value]
        gae = 0 
        advantages = []
        # 从后往前迭代整个 trajectory（即从 episode 最后一步倒着算），是 GAE 递归定义要求的。
        for t in reversed(range(len(self.buffer.rewards))):
            # 计算优势函数\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
            delta = self.buffer.rewards[t] + self.gamma * values[t+1] * (1 - self.buffer.dones[t]) - values[t]
            # 计算 GAE，根据 GAE 的递推公式更新当前步的优势值
            gae = delta + self.gamma * self.lam * (1 - self.buffer.dones[t]) * gae
            # 将当前 timestep 的 GAE 插入最前面
            advantages.insert (0, gae)
        return advantages

    def update(self):
        #向前传播；
        with torch.no_grad():
            # 取self.buffer.states这个列表中最后一个状态的，也就是最新的状态
            next_state = self.buffer.states[-1]
            # 调用神经网络对next_state进行一次向前传播，但是由于self.policy(next_state)返回一个(action_probs, state_value)，我们只需要state_value
            _, next_value = self.policy(next_state)

        # 计算优势函数
        advantage = self.computer_gae(next_value)
        # advantage是一个列表，表示每个时间步的 优势值 A(s, a）, zip(advantage, self.buffer.values) 把两个列表配对起来
        # 构造每一步的目标回报 R_t ≈ A_t + V(s_t)，用于更新 Critic 网络
        returns = [adv + val for adv, val in zip (advantage, self.buffer.values)]

        #转tensor
        # 把原本的列表变成一个二维张量，[tensor([1,2]), tensor([3,4])] → torch.stack → tensor([[1,2],[3,4]])
        states = torch.stack(self.buffer.states)
        # 将动作列表 [a1, a2, a3, ...] 转换为张量，用于后续计算
        actions = torch.tensor(self.buffer.actions)
        log_probs_old = torch.stack(self.buffer.log_probs)
        # detach() 的作用是从计算图中“分离”出来，防止梯度反向传播；
        returns = torch.tensor(returns).detach()
        advantage = torch.tensor(advantage).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) #normalize

        # PPO 多次epoch更新
        for _ in range(5):
            # 计算 Actor 网络和 Critic 网络的输出
            probs, values = self.policy(states)
            
            # 根据神经网络输出的动作概率 probs，创建一个 离散动作的分布对象
            # Categorical 是 PyTorch 的一个离散分布类，用于处理 多个动作中随机采样一个的情形；
            dist = torch.distributions.Categorical(probs)
            # 取出你执行的 actions 在当前策略下的对数概率
            log_probs = dist.log_prob(actions)
            
            # 计算 Surrogate Loss
            # 计算当前策略和旧策略对同一动作的概率之比。
            rations = torch.exp(log_probs - log_probs_old.detach())
            # PPO目标函数的第一项 L^{\text{PG}} = \mathbb{E} \left[ \text{ratio} \cdot \text{Advantage} \right]
            surr1 = rations * advantage
            # PPO裁剪目标函数的第二项 L^{\text{CLIP}} = \mathbb{E} \left[ \min \left( \text{ratio} \cdot A, \text{clipped\_ratio} \cdot A \right) \right]
            surr2 = torch.clamp(rations, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # 取两个 surrogate 中较小的那个（保守策略）
            # 加负号是因为我们要最大化 PPO目标
            actor_loss = -torch.min(surr1, surr2).mean()

            # 计算 Critic 网络的均方误差损失（MSE Loss）
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
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
    episode_reward = 0

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 存储奖励和终止标志
        agent.buffer.rewards.append(reward)
        agent.buffer.dones.append(done)
        episode_reward += reward

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
import gymnasium as gym
import torch
import torch.nn as nn

# 定义网络
class ActorCritic(nn.Module):
    def __init__(self,state_dim,action_dim, hidden_dim):
        super(ActorCritic,self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    def forward(self,x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value
    
# 环境初始化
env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0] # 状态维度
action_dim = env.action_space.n # 动作维度

#定义存储一批轨迹（buffer）
"""
	•	self.states: 存储所有时间步的状态 s_t
	•	self.actions: 存储所有时间步采取的动作 a_t
	•	self.log_probs: 存储所有时间步动作对应的对数概率，用于 PPO 的 ratio
	•	self.rewards: 存储所有时间步获得的奖励 r_t
	•	self.values: 存储 Critic 网络估算的状态价值 V(s_t)，用于计算优势值（Advantage）
	•	self.dones: 存储是否为 episode 结束（1 表示结束，0 表示未结束）
"""
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    # 清空 buffer 中所有轨迹数据，为下一个 rollout 做准备
    def clear(self):
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
        #送入神经网络当中获取策略
        self.policy = ActorCritic(state_dim,action_dim, hidden_dim)
        #为 PPO 算法中的 Actor 和 Critic 网络分别设置优化器参数，以便后续进行反向传播和梯度更新
        #self.policy.actor.parameters()：提取 actor 网络中所有可训练参数。actor 负责输出策略（即动作概率）
        #self.policy.critic.parameters()：提取 critic 网络中所有可训练参数。critic 负责估算状态价值
        self.optimizer = torch.optim.Adam([
            {'params':self.policy.actor.parameters(),'lr':lr_actor},
            {'params':self.policy.critic.parameters(),'lr':lr_critic}
        ])
        #折扣因子
        self.gamma = gamma 
        # GAE 参数
        self.lam = lam
        # PPO裁剪参数
        self.eps_clip = eps_clip
        self.buffer = RolloutBuffer()
    
    def select_action(self, state):
        # 从神经网络中选择动作，dtype=torch.float32是因为神经网络中强行要求输入的是float32的数据
        state = torch.tensor(state,dtype=torch.float32)
        # 向前传播，进行推理，得到动作概率和状态价值（）
        with torch.no_grad():
            probs, value = self.policy(state)
        # 从神经网络获取动作概率
        dist = torch.distributions.Categorical(probs)
        # 从动作概率中选择动作
        action = dist.sample()
        # 计算动作对应的对数概率（PPO 的核心是：比较新旧策略的比值 r = π_new / π_old，这个比就是通过对数计算的）
        log_prob = dist.log_prob(action)
        
        # 将状态，动作，对数概率，状态价值存入 buffer
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value)

        return action.item()

#计算优势函数
    def computer_gae(self, next_value):
        # self.buffer.values 是一个列表，包含当前 trajectory 中每个时间步的状态价值 V(s_t)
        # next_value 是最后一个状态的 下一个状态价值，用于 bootstrap（引导）最后一步
        values = self.buffer.values + [next_value]
        gae = 0 
        advantages = []
        # 从后往前迭代整个 trajectory（即从 episode 最后一步倒着算），是 GAE 递归定义要求的。
        for t in reversed(range(len(self.buffer.rewards))):
            # 计算优势函数\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
            delta = self.buffer.rewards[t] + self.gamma * values[t+1] * (1 - self.buffer.dones[t]) - values[t]
            # 计算 GAE，根据 GAE 的递推公式更新当前步的优势值
            gae = delta + self.gamma * self.lam * (1 - self.buffer.dones[t]) * gae
            # 将当前 timestep 的 GAE 插入最前面
            advantages.insert (0, gae)
        return advantages

    def update(self):
        #向前传播；
        with torch.no_grad():
            # 取self.buffer.states这个列表中最后一个状态的，也就是最新的状态
            next_state = self.buffer.states[-1]
            # 调用神经网络对next_state进行一次向前传播，但是由于self.policy(next_state)返回一个(action_probs, state_value)，我们只需要state_value
            _, next_value = self.policy(next_state)

        # 计算优势函数
        advantage = self.computer_gae(next_value)
        # advantage是一个列表，表示每个时间步的 优势值 A(s, a）, zip(advantage, self.buffer.values) 把两个列表配对起来
        # 构造每一步的目标回报 R_t ≈ A_t + V(s_t)，用于更新 Critic 网络
        returns = [adv + val for adv, val in zip (advantage, self.buffer.values)]

        #转tensor
        # 把原本的列表变成一个二维张量，[tensor([1,2]), tensor([3,4])] → torch.stack → tensor([[1,2],[3,4]])
        states = torch.stack(self.buffer.states)
        # 将动作列表 [a1, a2, a3, ...] 转换为张量，用于后续计算
        actions = torch.tensor(self.buffer.actions)
        log_probs_old = torch.stack(self.buffer.log_probs)
        # detach() 的作用是从计算图中“分离”出来，防止梯度反向传播；
        returns = torch.tensor(returns).detach()
        advantage = torch.tensor(advantage).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) #normalize

        # PPO 多次epoch更新
        for _ in range(5):
            # 计算 Actor 网络和 Critic 网络的输出
            probs, values = self.policy(states)
            
            # 根据神经网络输出的动作概率 probs，创建一个 离散动作的分布对象
            # Categorical 是 PyTorch 的一个离散分布类，用于处理 多个动作中随机采样一个的情形；
            dist = torch.distributions.Categorical(probs)
            # 取出你执行的 actions 在当前策略下的对数概率
            log_probs = dist.log_prob(actions)
            
            # 计算 Surrogate Loss
            # 计算当前策略和旧策略对同一动作的概率之比。
            rations = torch.exp(log_probs - log_probs_old.detach())
            # PPO目标函数的第一项 L^{\text{PG}} = \mathbb{E} \left[ \text{ratio} \cdot \text{Advantage} \right]
            surr1 = rations * advantage
            # PPO裁剪目标函数的第二项 L^{\text{CLIP}} = \mathbb{E} \left[ \min \left( \text{ratio} \cdot A, \text{clipped\_ratio} \cdot A \right) \right]
            surr2 = torch.clamp(rations, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # 取两个 surrogate 中较小的那个（保守策略）
            # 加负号是因为我们要最大化 PPO目标
            actor_loss = -torch.min(surr1, surr2).mean()

            # 计算 Critic 网络的均方误差损失（MSE Loss）
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
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
n_episodes = 100
max_steps = 1000  # 每回合最多多少步
update_timestep = 5000  # 每累计这么多步，更新一次
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
    episode_reward = 0

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 存储奖励和终止标志
        agent.buffer.rewards.append(reward)
        agent.buffer.dones.append(done)
        episode_reward += reward

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


def test_agent(agent, env, num_episodes=100):
    """Test PPO agent performance without learning."""
    total_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

# Run test
test_agent(agent, env)


        



    


        
        
        

