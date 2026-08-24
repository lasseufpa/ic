from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
)

model.learn(total_timesteps=100_000)

model.save("ppo_cartpole")
