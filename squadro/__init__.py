import logging
import os

from squadro.animation.animated_game import GamePlay, GameAnimation
from squadro.game import Game
from squadro.tools.agents import AVAILABLE_AGENTS
from squadro.tools.log import logger
from squadro.training.deep_q_learning import DeepQLearningTrainer
from squadro.training.q_learning import QLearningTrainer

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = '1'

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
