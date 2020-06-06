#! python3

import os, time, glob, re
import gym
import gym_snake_game

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN

## -----------------------------------------------------------------------
##                                                    Learning parameters
##                                                   ---------------------

# NOTE if you use this program, you should set value of this params first


MODE = "LOAD" # LOAD or LEARNING or LOAD_AND_LEARNING

ITERATION_NUMS = 100000000
ITERATIONS_PER_SAVE = 1000000

SAVE_DIR = "models/dqn_snake_game_10x10"
SAVE_FILE_NAME = "iter_"
SAVE_FILE_PATH = os.path.join(SAVE_DIR, SAVE_FILE_NAME)

REPLAY_ITERATION_NUMS = 1000
REPLAY_INTERVAL_MS = 100
REPLAY_MP4_NAME = 'animation10x10_itermax.mp4'

LOG_DIR = './logs10x10_dqn/'

## -----------------------------------------------------------------------
##                                                    Parameter assertion
##                                                   ---------------------
assert MODE in ["LOAD", "LEARNING", "LOAD_AND_LEARNING"], \
                    "invalid mode : allowed mode is ['LOAD', 'LEARNING', 'LOAD_AND_LEARNING']"
assert 1000 < ITERATION_NUMS, \
                    "learning iteration should be more than 1000."
assert ITERATIONS_PER_SAVE < ITERATION_NUMS, \
                    "save iterations should be less than ITERATION_NUMS"

## -----------------------------------------------------------------------
##                                                    make directories
##                                                   ---------------------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

## -----------------------------------------------------------------------
##                                                    main learning
##                                                   ---------------------
env = gym.make('SnakeGame-v0')
env = Monitor(env, LOG_DIR, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

## MODEL LOADING
last_learning_iter = 0
model = None
if MODE in ["LOAD", "LOAD_AND_LEARNING"]:
    last_model_path = sorted(glob.glob(os.path.join(SAVE_DIR, "*")), key=lambda x:int((re.search(r"iter_(\d+)", x)).group(1)))[-1]
    match = re.search(r"iter_(\d+)", last_model_path)
    last_learning_iter = int(match.group(1))
    model = PPO2.load(last_model_path, tensorboard_log=LOG_DIR)
    # model = DQN.load(last_model_path, tensorboard_log=LOG_DIR)
    model.set_env(env)

## MODEL LEARNING
save_iterations = [0]
for x in range(2, 8):
     if 10**x < ITERATIONS_PER_SAVE:
         save_iterations.append(10**x)
x = ITERATIONS_PER_SAVE
while x < ITERATION_NUMS:
    save_iterations.append(x)
    x += ITERATIONS_PER_SAVE
save_iterations.append(ITERATION_NUMS)

print(save_iterations)

if MODE in ["LEARNING", "LOAD_AND_LEARNING"]:
    if model is None:
        model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=LOG_DIR)
        # model = DQN(MlpPolicy, env, verbose=1, tensorboard_log=LOG_DIR)
        print("### model initialized")

    for li in range(1, len(save_iterations)):
        current_learning_iter = None
        if save_iterations[li] < last_learning_iter:
            continue
        else:
            current_learning_iter = save_iterations[li] - min(last_learning_iter, save_iterations[li-1])

        model.learn(total_timesteps=current_learning_iter, reset_num_timesteps=False)
        model.save(SAVE_FILE_PATH + str(save_iterations[li]))

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

ims = []

observation = env.reset()
for i in range(REPLAY_ITERATION_NUMS):
    img = env.render()
    im = plt.imshow(img)
    ims.append([im])

    ## for realtime drawing
    # plt.draw()
    # plt.pause(0.00001)
    # plt.cla() 
    
    action, _ = model.predict(observation)
    observation, reward, done, info = env.step(action)

    if done:
        env.reset()

# for mp4
ani = animation.ArtistAnimation(fig, ims, interval=REPLAY_INTERVAL_MS, repeat=False, blit=True)
ani.save(REPLAY_MP4_NAME, writer="ffmpeg", dpi=300)
plt.show()