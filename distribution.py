#! python3

import os, time
import gym
import gym_snake_game

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

env = gym.make('SnakeGame-v0')

observation = env.reset()

hist = np.zeros((5, 5))

for i in range(1000000):
    x, y = np.where(observation==1)
    if len(x) != 0:
        x, y = x[0], y[0]
        hist[y][x] += 1

    # img = env.render()
    # plt.imshow(img)

    # plt.draw()
    # plt.pause(0.00001)
    # plt.cla() 

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        env.reset()

hist = hist 
plt.imshow(hist)
plt.colorbar()
plt.savefig('distribution_5x5.png')
plt.show()

