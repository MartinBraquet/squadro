from squadro.agents.alphabeta_agent import AlphaBetaAdvancementDeepAgent
from squadro.tools.constants import DefaultParams


def get_best_agent(**kwargs):
    return AlphaBetaAdvancementDeepAgent(**kwargs)


def get_best_real_time_game_agent(**kwargs):
    kwargs.setdefault('max_time_per_move', DefaultParams.max_time_per_move_real_time)
    return get_best_agent(**kwargs)
