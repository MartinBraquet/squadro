from squadro.agents.agent import Agent
from squadro.agents.alphabeta_agent import (
    AlphaBetaAdvancementAgent,
    AlphaBetaRelativeAdvancementAgent,
    AlphaBetaAdvancementDeepAgent
)
from squadro.agents.human_agent import HumanAgent
from squadro.agents.random_agent import RandomAgent

AGENTS = (
    HumanAgent,
    RandomAgent,
    AlphaBetaAdvancementAgent,
    AlphaBetaRelativeAdvancementAgent,
    AlphaBetaAdvancementDeepAgent,
)
AGENTS = {a.get_name(): a for a in AGENTS}
AVAILABLE_AGENTS = list(AGENTS.keys())


def get_agent(agent, **kwargs):
    if isinstance(agent, Agent):
        return agent
    agent = agent.replace('_agent', '')
    return AGENTS[agent](**kwargs)
