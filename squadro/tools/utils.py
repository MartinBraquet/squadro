from squadro import human_agent, smart_agent, random_agent, basic_agent

AGENTS = {
    'human': human_agent,
    'smart': smart_agent,
    'random': random_agent,
    'basic': basic_agent,
}


def get_agent(agent_name):
    agent_name = agent_name.replace('_agent', '')
    return getattr(AGENTS[agent_name], 'MyAgent')()
