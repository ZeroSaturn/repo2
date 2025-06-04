import os
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .env import DoublePendulumEnv
from .ddpg import DDPGAgent, ReplayBuffer


def render(env, ax, state):
    th1 = np.arctan2(state[1], state[0])
    th2 = np.arctan2(state[3], state[2]) - th1
    origin = np.array([0, 0])
    p1 = origin + np.array([np.sin(th1), -np.cos(th1)])
    p2 = p1 + np.array([np.sin(th1 + th2), -np.cos(th1 + th2)])
    ax.clear()
    ax.plot([origin[0], p1[0]], [origin[1], p1[1]], 'k-', lw=2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Double Pendulum Balance')


def train(num_episodes=200):
    env = DoublePendulumEnv()
    obs_dim = 6
    act_dim = 1
    agent = DDPGAgent(obs_dim, act_dim, act_limit=env.max_torque)
    replay = ReplayBuffer()

    fig, ax = plt.subplots()
    rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        for step in range(200):
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            replay.add(obs, action, reward, next_obs)
            obs = next_obs
            episode_reward += reward

            if len(replay.states) > 1000:
                agent.train(replay)

            if step % 5 == 0:
                render(env, ax, obs)
                plt.pause(0.001)

        rewards.append(episode_reward)
        print(f"Episode {episode}: reward {episode_reward:.2f}")

    plt.figure()
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


if __name__ == "__main__":
    train()
