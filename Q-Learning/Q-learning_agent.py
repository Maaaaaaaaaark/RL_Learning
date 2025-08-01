from collections import defaultdict
import gymnasium as gym
import numpy as np

class BlackjackAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            discount_factor: float,
            final_epsilon: float,
    ):
       """
       初始化一个Q学习代理。
        参数：
            env：训练环境
            learning_rate：更新Q值的速度（0-1）
            initial_epsilon：初始探索率（通常为1.0）
            epsilon_decay：每个回合减少epsilon的幅度
            final_epsilon：最小探索率（通常为0.1）
            discount_factor：未来奖励的价值系数（0-1）
       """ 
       self.env = env

        # Q表：将状态和动作映射到预期奖励
        # defaultdict会自动为新状态创建值为0的条目
       self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

       self.lr = learning_rate
       self.discount_factor = discount_factor

       # Exploration parameters
       self.epsilon = initial_epsilon
       self.epsilon_decay = epsilon_decay
       self.final_epsilon = final_epsilon

       # 跟踪学习进度
       self.training_error = []
    def get_action(self, obs:[int, int, bool]) -> int:
        """
        使用epsilon-greedy选择一个策略
        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # 以概率 ε：探索（随机动作）
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # 以概率 1-ε：选择最优动作
        else:
            return int(np.argmax(self.q_values[obs]))
    
    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            next_obs: tuple[int, int, bool],
            done: bool,
    ):
        """根据经验更新 Q 值。
        这是 Q 学习的核心：从 (状态、动作、奖励、下一个状态) 中学习。
        """
        # 从下一个状态出发，我们能做到的最好结果是什么？
        #(如果剧集终止，则为零——未来无法获得奖励）
        future_q_values = (not terminated) * np.max(self.q_values[next_obs])

        # bellman equation
        target = reward + self.discount_factor * future_q_values

        # 我们的当前估算有多不准确？
        temporal_difference = target - self.q_values[obs][action]

        # 将我们的估计值向误差方向更新。
        # 学习率控制我们迈出的步长大小。
        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_difference)

        # 跟踪学习进度（用于调试）
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        # 每次事件后降低探索速率。
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


learning_rate = 0.01
n_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

# 初始化环境和agent
env = gym.make("Blackjack-v1", sab = False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length = n_episodes)

agent = BlackjackAgent(
    env = env,
    learning_rate = learning_rate,
    initial_epsilon = start_epsilon,
    epsilon_decay = epsilon_decay,
    final_epsilon = final_epsilon,
    discount_factor= 0.9
)

from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    # 重置环境
    obs, info = env.reset()
    done = False

    # 执行一个完整的回合
    while not done:
        # 选择动作
        action = agent.get_action(obs)

        # 执行动作并观察结果
        next_obs, reward, terminated, truncated, info = env.step(action)

        # 从经验中学习
        agent.update(obs, action, reward, next_obs, terminated)

        # Move to next state
        done = terminated or truncated
        obs = next_obs

    # 降低探索速率
    agent.decay_epsilon()



#####
# Test the trained agent
def test_agent(agent, env, num_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

# Test your agent
test_agent(agent, env)