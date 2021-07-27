from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from wrappers import wrapper
from gym.wrappers import Monitor

import keras as k
import numpy as np

RENDER_REALTIME = True

# Build env (first level, right only)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper(env)
if not RENDER_REALTIME:
    env = Monitor(env, './videos/v5', force=True)

# Parameters
states = (84, 84, 4)
actions = env.action_space.n

model = k.models.load_model("./backupmodels/v4.1")

state = env.reset()
total_reward = 0
while True:
    action = np.argmax(model.predict(np.array([state])))
    state, reward, done, _ = env.step(action)
    total_reward += reward
    if RENDER_REALTIME:
        env.render()
    if done:
        break
print(total_reward)

# 2
