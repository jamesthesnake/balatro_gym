import gymnasium as gym
from gymnasium import spaces
import numpy as np

import gymnasium as gym, balatro_gym as bg
env = bg.make("EightCardDraw-v0")
obs, _ = env.reset()
done = False
while not done:
    a = np.random.choice(np.nonzero(obs["action_mask"])[0])
    obs, reward, done, trunc, info = env.step(a)
print("reward:", reward)

