import gymnasium as gym
import numpy as np
from collections import defaultdict

class FrozenLakeAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            discount_factor: float,
            final_epsilon: float,
            ):
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

        # 跟踪学习进度（用于调试）
        self.training_error = []

    def decay_epsilon(self):
        # 每次事件后降低探索速率。
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            next_obs: tuple[int, int, bool],
            done: bool,
    ):
        future_q_values = (not terminated) * np.max(self.q_values[next_obs])

        target = reward + self.discount_factor * future_q_values

        temporal_difference = target - self.q_values[obs][action]

        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_difference)

        self.training_error.append(temporal_difference)
    
    def get_action(self,obs):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

learning_rate = 0.01
n_episodes = 1000000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

env = gym.make("FrozenLake-v1",render_mode="rgb_array")

agent = FrozenLakeAgent(
    env = env,
    learning_rate = learning_rate,
    initial_epsilon = start_epsilon,
    epsilon_decay = epsilon_decay,
    final_epsilon = final_epsilon,
    discount_factor = 0.9
)

from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done =  False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, next_obs, terminated)
        agent.decay_epsilon()
        obs = next_obs
        done = terminated or truncated

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
        