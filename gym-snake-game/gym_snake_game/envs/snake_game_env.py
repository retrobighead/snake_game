
import gym

import numpy as np
from PIL import Image
from collections import deque

import time

import logging
logger = logging.getLogger(__name__)


class Screen:
        def __init__(self, screen_width, screen_height):

            self.height = screen_height
            self.width = screen_width
            self.channels = 3

            self.colors = {
                "white": np.array([255, 255, 255]),
                "light_gray": np.array([200, 200, 200]),
                "green": np.array([38, 166, 154]),
                "light_light_gray": np.array([224, 224, 224]),
                "red": np.array([255, 138, 101]),
                "yellow": np.array([253, 216, 53])
            }

            self.pixels = None
            self.init_pixels()
        
        def init_pixels(self):
            self.pixels = np.full((self.height, self.width, self.channels), 255)

        def fill(self, color_label="white"):
            col = self.colors.get(color_label)
            self.pixels[:self.height, :self.width] = col

        def rect(self, color_label, x_start, y_start, width, height):
            col = self.colors.get(color_label)
            self.pixels[y_start:y_start+height, x_start:x_start+width] = col
        
        def line_x(self, color_label, x, start_y, end_y, line_width=3):
            col = self.colors.get(color_label)
            self.pixels[start_y:end_y+1, x - line_width//2:x - line_width//2 + line_width] = col
        
        def line_y(self, color_label, y, start_x, end_x, line_width=3):
            col = self.colors.get(color_label)
            self.pixels[y - line_width//2:y - line_width//2 + line_width, start_x:end_x] = col
            
        def draw_base_lines(self, color_label, padding, grid_num_horizontal, grid_num_vertical, grid_width, line_width=3):
            for x in range(grid_num_horizontal + 1):
                self.line_x(color_label, padding + grid_width*x, padding, padding + grid_num_horizontal*grid_width, line_width)
            for y in range(grid_num_vertical + 1):
                self.line_y(color_label, padding + grid_width*y, padding, padding + grid_num_vertical*grid_width, line_width)
            

class SnakeGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        self.caption_title = 'Snake Game RL | RBH'
        self.screen_width = 1080
        self.screen_height = 1080

        self.grid_num_w = 10
        self.grid_num_h = 10

        self.padding = 100
        self.grid_width = 88

        self.label_bg = 0
        self.label_snake_head = 1
        self.label_snake_body = 2
        self.label_snake_tail = 3
        self.label_feed = 4
        self.label_wall = 5

        self.color_bg = "white"
        self.color_border = "light_gray"
        self.color_snake_head = "green"
        self.color_snake_body = "light_light_gray"
        self.color_snake_tail = "red"
        self.color_feed = "yellow"

        self.reset()

        self.screen = Screen(self.screen_width, self.screen_height)

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=5,
            shape=(5, 5)
        )
        self.reward_range = [-10., 100.]

    def step(self, action):

        if self.done:
            return None, None, None, None

        reward = 0

        if action == 0 and self.snake_direction_labels[self.snake_direction] != 'down':
            self.snake_direction = 0
        if action == 1 and self.snake_direction_labels[self.snake_direction] != 'left':
            self.snake_direction = 1
        if action == 2 and self.snake_direction_labels[self.snake_direction] != 'up':
            self.snake_direction = 2
        if action == 3 and self.snake_direction_labels[self.snake_direction] != 'right':
            self.snake_direction = 3

        self.snake_x += self.snake_directions[self.snake_direction][0]
        self.snake_y += self.snake_directions[self.snake_direction][1]
        self.snake_positions.appendleft((self.snake_x, self.snake_y))

        if self.judge_gameover():
            self.snake_positions.popleft()
            reward = -10
            self.done = True

        if self.feed_x == self.snake_x and self.feed_y == self.snake_y:
            self.feed_exist = False
            self.place_feed()
            reward = 10
            if len(self.snake_positions) == self.grid_num_w * self.grid_num_h:
                reward = 100
        else:
            self.snake_positions.pop()

        self.stage = np.full((self.grid_num_h, self.grid_num_w), self.label_bg)
        self.stage[self.feed_y][self.feed_x] = self.label_feed
        for i in range(len(self.snake_positions)):
            x, y = self.snake_positions[i]
            self.stage[y][x] = self.label_snake_body
        self.stage[self.snake_positions[0][1]][self.snake_positions[0][0]] = self.label_snake_head
        self.stage[self.snake_positions[-1][1]][self.snake_positions[-1][0]] = self.label_snake_tail

        observation = self.observe()
        return observation, reward, self.done, {}

    def reset(self):
        self.done = False

        self.stage = np.full((self.grid_num_h, self.grid_num_w), self.label_bg)

        self.snake_x = np.random.randint(2, self.grid_num_w-2)
        self.snake_y = np.random.randint(2, self.grid_num_h-2)
        self.snake_direction = np.random.randint(0, 4)
        self.snake_directions = [(0, -1), (1, 0), (0, 1), (-1, 0)] # x, y
        self.snake_direction_labels = ['up', 'right', 'down', 'left']
        tail_x = self.snake_x - self.snake_directions[self.snake_direction][0] # x
        tail_y = self.snake_y - self.snake_directions[self.snake_direction][1] # y

        self.snake_positions = deque([]) # x, y
        self.snake_positions.append((self.snake_x, self.snake_y))
        self.snake_positions.append((tail_x, tail_y))

        self.stage[self.snake_y][self.snake_x] = self.label_snake_head
        self.stage[tail_y][tail_x] = self.label_snake_tail

        self.feed_exist = False
        self.feed_x = None
        self.feed_y = None
        self.place_feed()

        return self.observe()

    def place_feed(self):
        if not self.feed_exist:
            ys, xs = np.where(self.stage == self.label_bg)
            ind = np.random.randint(ys.shape[0])
            self.feed_y, self.feed_x = ys[ind], xs[ind]
            self.feed_exist = True
            self.stage[self.feed_y][self.feed_x] = self.label_feed

    def judge_gameover(self):
        if self.snake_x < 0 or self.grid_num_w <= self.snake_x:
            return True
        if self.snake_y < 0 or self.grid_num_h <= self.snake_y:
            return True

        if self.stage[self.snake_y][self.snake_x] == self.label_snake_body:
            return True
        if self.stage[self.snake_y][self.snake_x] == self.label_snake_tail:
            return True

        return False

    def observe(self):
        ret = np.full((5, 5), self.label_wall)
        for x in range(max(0, self.snake_x-2), min(self.grid_num_w, self.snake_x+3)):
            for y in range(max(0, self.snake_y-2), min(self.grid_num_h, self.snake_y+3)):
                ret[y - self.snake_y + 2][x - self.snake_x + 2] = self.stage[y][x]
        return ret

    def render(self, mode="human"):

        self.screen.fill(self.color_bg)

        for y in range(self.grid_num_h):
            for x in range(self.grid_num_w):
                x_start, y_start = self.padding + self.grid_width*x, self.padding + self.grid_width*y

                if self.stage[y][x] == self.label_snake_head:
                    self.screen.rect(self.color_snake_head, x_start, y_start, self.grid_width, self.grid_width)
                elif self.stage[y][x] == self.label_snake_body:
                    self.screen.rect(self.color_snake_body, x_start, y_start, self.grid_width, self.grid_width)
                elif self.stage[y][x] == self.label_snake_tail:
                    self.screen.rect(self.color_snake_tail, x_start, y_start, self.grid_width, self.grid_width)
                elif self.stage[y][x] == self.label_feed:
                    self.screen.rect(self.color_feed, x_start, y_start, self.grid_width, self.grid_width)
        
        self.screen.draw_base_lines(self.color_border, self.padding, self.grid_num_w, self.grid_num_h, self.grid_width)

        return self.screen.pixels

    def close(self):
        pass
