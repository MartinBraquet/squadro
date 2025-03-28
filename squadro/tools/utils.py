from squadro import human_agent, smart_agent, random_agent, basic_agent
from squadro.agent import Agent

AGENTS = {
    'human': human_agent,
    'smart': smart_agent,
    'random': random_agent,
    'basic': basic_agent,
}
AVAILABLE_AGENTS = list(AGENTS.keys())


def get_agent(agent):
    if isinstance(agent, Agent):
        return agent
    agent = agent.replace('_agent', '')
    return getattr(AGENTS[agent], 'MyAgent')()
