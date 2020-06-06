import pyxel
import itertools
import numpy as np
from collections import deque


class App:
    def __init__(self):
        self.game_state = 0
        self.game_states = ['starting', 'playing', 'game_over']
        self.trial = 0

        self.grid_size = 10
        self.grid_num_v = 15
        self.grid_num_h = 15
        self.text_area_width = 50
        self.padding = 10

        self.color_bg = 0
        self.color_border = 13
        self.color_snake_head = 3
        self.color_snake_body = 7
        self.color_snake_tail = 14
        self.color_feed = 10
        self.color_text = 7
        self.color_arrow_up = 7
        self.color_arrow_right = 7
        self.color_arrow_down = 7
        self.color_arrow_left = 7
        self.color_refresh_rate = 5

        self.width = self.grid_size * self.grid_num_h + self.text_area_width + self.padding*2
        self.height = self.grid_size * self.grid_num_v + self.padding*2

        pyxel.init(self.width, self.height, fps=15)
        self.reset()
        pyxel.run(self.update, self.draw)

    def reset(self):
        self.snake_x = np.random.randint(1, self.grid_num_v-1)
        self.snake_y = np.random.randint(1, self.grid_num_h-1)
        self.snake_length = 2
        self.snake_direction = 0
        self.snake_direction_labels = ['up', 'right', 'down', 'left']
        self.snake_direction_movs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.snake_positions = deque([])
        self.snake_positions.append((self.snake_y, self.snake_x))
        mov = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        mov_x, mov_y = mov[np.random.randint(4)]
        self.snake_positions.append((self.snake_y + mov_y, self.snake_x + mov_x))

        self.feed_x = np.random.choice([x for x in range(self.grid_num_h) if x != self.snake_positions[0][1] and x != self.snake_positions[1][1]])
        self.feed_y = np.random.choice([y for y in range(self.grid_num_v) if y != self.snake_positions[0][0] and y != self.snake_positions[1][0]])
        self.feed_exist = True

        self.stage = np.full((self.grid_num_v, self.grid_num_h), self.color_bg)
        self.stage[self.snake_positions[0][0]][self.snake_positions[0][1]] = self.color_snake_head
        self.stage[self.snake_positions[-1][0]][self.snake_positions[-1][1]] = self.color_snake_tail
        self.stage[self.feed_y][self.feed_x] = self.color_feed_value()

        self.trial += 1

    def increment_state(self):
        self.game_state = (self.game_state + 1)%len(self.game_states)

    def update(self):
        if self.game_states[self.game_state] == 'starting':
            if pyxel.btn(pyxel.KEY_SPACE):
                self.increment_state()
        if self.game_states[self.game_state] == 'playing':
            self.update_while_playing()
            if self.judge_gameover():
                self.increment_state()
        if self.game_states[self.game_state] == 'game_over':
            if pyxel.btn(pyxel.KEY_SPACE):
                self.increment_state()
                self.reset()

    def judge_gameover(self):
        if self.snake_x < 0:
            return True
        if self.grid_num_h <= self.snake_x:
            return True
        if self.snake_y < 0:
            return True
        if self.grid_num_v <= self.snake_y:
            return True

        for pos in itertools.islice(self.snake_positions, 1, len(self.snake_positions)):
            if pos == (self.snake_y, self.snake_x):
                return True

        return False

    def color_feed_value(self):
        if pyxel.frame_count % self.color_refresh_rate != 0:
            return self.color_feed

        if self.color_feed == 10:
            self.color_feed = 9
        else:
            self.color_feed = 10
        return self.color_feed

    def color_head_value(self):
        if pyxel.frame_count % self.color_refresh_rate != 0:
            return self.color_snake_head

        if self.color_snake_head == 11:
            self.color_snake_head = 3
        else:
            self.color_snake_head = 11
        return self.color_snake_head

    def color_tail_value(self):
        if pyxel.frame_count % self.color_refresh_rate != 0:
            return self.color_snake_tail

        if self.color_snake_tail == 15:
            self.color_snake_tail = 14
        else:
            self.color_snake_tail = 15
        return self.color_snake_tail

    def draw(self):
        pyxel.cls(self.color_bg)

        # line
        for x in range(0, self.grid_num_h+1):
            _x = x*self.grid_size + self.padding
            pyxel.line(_x, self.padding, _x, self.padding + self.grid_size*self.grid_num_v, self.color_border)
        for y in range(0, self.grid_num_v+1):
            _y = y*self.grid_size + self.padding
            pyxel.line(self.padding, _y, self.padding + self.grid_size*self.grid_num_h, _y, self.color_border)

        # snake
        for y in range(self.grid_num_v):
            for x in range(self.grid_num_h):
                _x, _y = self.padding + x*self.grid_size + 1, self.padding + y*self.grid_size + 1
                width, height = self.grid_size-1, self.grid_size-1                    
                pyxel.rect(_x, _y, width, height, self.stage[y][x])
        
        # snake joint area
        for i in range(len(self.snake_positions)-1):
            y1, x1 = self.snake_positions[i]
            y2, x2 = self.snake_positions[i+1]
            if x1 == x2:
                _x = x1*self.grid_size + self.padding+1
                _y = max(y1, y2)*self.grid_size + self.padding
                pyxel.line(_x, _y, _x+self.grid_size-2, _y, self.color_snake_body)
            if y1 == y2: 
                _x = max(x1, x2)*self.grid_size + self.padding
                _y = y1*self.grid_size + self.padding+1
                pyxel.line(_x, _y, _x, _y+self.grid_size-2, self.color_snake_body)
            
        # text area
        x_left = self.padding + self.grid_size * self.grid_num_h + self.padding
        pyxel.text(x_left, 10, '-- state --', 13)
        pyxel.text(x_left, 20, self.game_states[self.game_state], self.color_text)
        pyxel.text(x_left, 40, '-- param --', 13)
        pyxel.text(x_left, 50, 'trial:' + str(self.trial), self.color_text)
        pyxel.text(x_left, 60, 'length:' + str(self.snake_length), self.color_text)

        # draw triangles
        pyxel.text(x_left, 85, '-- input --', 13)
        pyxel.text(x_left, 95, 'key:' + self.snake_direction_labels[self.snake_direction], self.color_text)
        pyxel.tri(190, 108, 197, 120, 183, 120, self.color_arrow_up)
        pyxel.tri(168, 130, 180, 137, 180, 123, self.color_arrow_left)
        pyxel.tri(212, 130, 200, 137, 200, 123, self.color_arrow_right)
        pyxel.tri(190, 152, 197, 140, 183, 140, self.color_arrow_down)

    def update_arrow_color(self):
        self.color_arrow_up = 13
        self.color_arrow_right = 13
        self.color_arrow_down = 13
        self.color_arrow_left = 13

        if self.snake_direction_labels[self.snake_direction] == 'up':
            self.color_arrow_up = 10
        if self.snake_direction_labels[self.snake_direction] == 'right':
            self.color_arrow_right = 10
        if self.snake_direction_labels[self.snake_direction] == 'down':
            self.color_arrow_down = 10
        if self.snake_direction_labels[self.snake_direction] == 'left':
            self.color_arrow_left = 10

    def update_while_playing(self):
        # feed place
        if not self.feed_exist:
            ys, xs = np.where(self.stage == self.color_bg)
            ind = np.random.randint(ys.shape[0])
            self.feed_y, self.feed_x = ys[ind], xs[ind]
            self.feed_exist = True

        # snake direction
        if pyxel.btn(pyxel.KEY_RIGHT) and not self.snake_direction == 3:
            self.snake_direction = 1
        if pyxel.btn(pyxel.KEY_LEFT) and not self.snake_direction == 1:
            self.snake_direction = 3
        if pyxel.btn(pyxel.KEY_UP) and not self.snake_direction == 2:
            self.snake_direction = 0
        if pyxel.btn(pyxel.KEY_DOWN) and not self.snake_direction == 0:
            self.snake_direction = 2
        self.update_arrow_color()

        # moving snake head position
        self.snake_y = self.snake_y + self.snake_direction_movs[self.snake_direction][0]
        self.snake_x = self.snake_x + self.snake_direction_movs[self.snake_direction][1]

        # judge if getting feed
        if self.feed_x == self.snake_x and self.feed_y == self.snake_y:
            self.feed_exist = False
            self.snake_length += 1

        # adjust snake length
        if 0 <= self.snake_x and self.snake_x < self.grid_num_h and 0 <= self.snake_y and self.snake_y < self.grid_num_v:
            self.snake_positions.appendleft((self.snake_y, self.snake_x))
        if self.feed_exist:
            self.snake_positions.pop()

        # updage stage
        self.stage = np.full((self.grid_num_v, self.grid_num_h), self.color_bg)
        self.stage[self.feed_y][self.feed_x] = self.color_feed_value()
        for i in range(len(self.snake_positions)):
            x, y = self.snake_positions[i]
            self.stage[x][y] = self.color_snake_body
        self.stage[self.snake_positions[0][0]][self.snake_positions[0][1]] = self.color_snake_head
        self.stage[self.snake_positions[-1][0]][self.snake_positions[-1][1]] = self.color_snake_tail

app = App()
