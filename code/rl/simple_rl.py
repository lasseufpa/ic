import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

obs, info = env.reset()
num_actions_before_done = 0
for _ in range(5000):
    action = env.action_space.sample()  # random
    obs, reward, terminated, truncated, info = env.step(action)
    print("Reward received: ", reward, "action: ", action)
    num_actions_before_done += 1
    if terminated or truncated:
        print(f"Episode finished after {num_actions_before_done} actions")
        cart_position = obs[0]
        print(f"Failed! Cart position = {cart_position:.2f}")
        angle = obs[2]  # pole angle in radians
        print(f"Failed! Pole angle = {angle:.2f} rad")    
        obs, info = env.reset()
        num_actions_before_done = 0
env.close()
