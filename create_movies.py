#! python3

import os, time, glob, re
import gym
import gym_snake_game

from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

## -----------------------------------------------------------------------
##                                                              Functions
##                                                   ---------------------

def get_model_iteration(path):
    pat = re.compile(re.escape(os.path.join(load_dir_path, file_prefix)) + r'(\d+)\.zip')
    mo = pat.match(path)
    return mo.group(1)

## -----------------------------------------------------------------------
##                                                              Parameter
##                                                   ---------------------

load_dir_path = "./ppo2_snake_game_5x5"
file_prefix = "iter_"

save_dir = "./ingredients/"

## -----------------------------------------------------------------------
##                                                   Parameter Assertion
##                                                 -----------------------
os.makedirs("ingredients", exist_ok=True)
os.makedirs("ingredients/", exist_ok=True)

assert os.path.exists(load_dir_path), "load_dir_path does not exist"

## -----------------------------------------------------------------------
##                                                   Rendering
##                                                 -----------------------

# setup environment
env = gym.make('SnakeGame-v0')
env = DummyVecEnv([lambda: env])

# model list
model_files = glob.glob(os.path.join(load_dir_path, file_prefix + "*.zip"))
model_files = sorted(model_files, key=get_model_iteration)

## matplotlib animation
fig = plt.figure()
plt.tick_params(bottom=False,
                labelbottom=False,
                left=False,
                labelleft=False,
                right=False,
                labelright=False,
                top=False,
                labeltop=False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

for mf in model_files:
    iteration = get_model_iteration(mf)
    model = PPO2.load(mf)

    ims = []
    observation = env.reset()
    
    done_count = 0
    while True:
        img = env.render()
        im = plt.imshow(img)
        ims.append([im])

        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)

        if done:
            done_count += 1