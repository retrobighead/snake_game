import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='SnakeGame-v0',
    entry_point='gym_snake_game.envs:SnakeGameEnv'
)
