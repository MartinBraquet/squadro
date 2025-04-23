import os

from squadro.animation.animated_game import RealTimeAnimatedGame, GameAnimation
from squadro.game import Game
from squadro.tools.agents import AVAILABLE_AGENTS
from squadro.tools.log import logger
from squadro.training.q_learning import QLearningTrainer

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = '1'
