import importlib
import inspect
import pkgutil

from squadro.agents.agent import Agent


def list_agent_classes():
    """
    Lists all the classes in the `squadro.agents.agent` package that inherit from the `Agent` class.
    """
    agent_classes = []
    package_name = "squadro.agents"

    # Iterate through all modules in the squadro.agents package
    for _, module_name, is_pkg in pkgutil.iter_modules(
        importlib.import_module(package_name).__path__, package_name + "."
    ):
        if not is_pkg:  # Only process modules, not sub-packages
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if the class inherits from Agent and is not the Agent class itself
                if issubclass(obj, Agent) and obj is not Agent and not inspect.isabstract(obj):
                    agent_classes.append(obj)

    return agent_classes


AGENTS = list_agent_classes()
AGENTS = {a.get_name(): a for a in AGENTS}
AVAILABLE_AGENTS = list(AGENTS.keys())


def get_agent(agent, **kwargs):
    if isinstance(agent, Agent):
        return agent
    agent = agent.replace('_agent', '')
    if agent not in AGENTS:
        raise ValueError(f"Agent '{agent}' not found. Available agents: {AVAILABLE_AGENTS}")
    return AGENTS[agent](**kwargs)
