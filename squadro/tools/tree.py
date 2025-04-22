import json
import os
from collections import defaultdict

from squadro.state import State
from squadro.tools.log import monte_carlo_logger as logger


class Node:
    def __init__(self, state: State, depth: int = 0):
        self.state = state
        self.edges = []
        self.depth = depth

    def is_leaf(self) -> bool:
        return len(self.edges) == 0

    @property
    def player_turn(self) -> int:
        return self.state.get_cur_player()

    def __repr__(self) -> str:
        return repr(self.state)

    def get_edge_stats(self, to_string: bool = False) -> list | str:
        stats = [edge.stats for edge in self.edges]
        if to_string:
            stats = '\n'.join(str(s) for s in stats)
        return stats

    @property
    def children(self) -> list['Node']:
        return [edge.out_node for edge in self.edges]


def save_edge_values(d: dict, node: Node) -> None:
    for edge in node.edges:
        value = edge.stats.N
        idx = (edge.in_node.tree_index, edge.out_node.tree_index)
        d[idx] = value
        save_edge_values(d, edge.out_node)


class Debug:
    tree_wanted = False
    nodes = defaultdict(dict)
    node_counter = 0

    @classmethod
    def save_tree(cls, node: Node) -> None:
        if not cls.tree_wanted:
            return
        if not os.path.exists('results'):
            os.mkdir('results')

        with open('results/edges.json', 'w') as f:
            json.dump(cls.edges, f, indent=4)

        edge_values = {}
        save_edge_values(edge_values, node)
        edge_values = {str(key): value for key, value in edge_values.items()}
        with open('results/edge_values.json', 'w') as f:
            json.dump(edge_values, f, indent=4)

        with open('results/nodes.json', 'w') as f:
            json.dump(cls.nodes, f, indent=4)

        nested_nodes = get_nested_nodes(node)
        with open('results/nested_nodes.json', 'w') as f:
            json.dump(nested_nodes, f, indent=4)

    @classmethod
    def clear(cls, node: Node) -> None:
        if not cls.tree_wanted:
            return
        cls.edges = []
        cls.nodes = defaultdict(dict)
        cls.node_counter = 0
        if hasattr(node, 'tree_index'):
            del node.tree_index
        cls.save_node(node)

    @classmethod
    def save_node(cls, node: Node) -> None:
        if not cls.tree_wanted:
            return
        if not hasattr(node, 'tree_index'):
            node.tree_index = cls.node_counter
            cls.node_counter += 1
        cls.nodes[node.tree_index] |= {
            # 'eval': eval_type,
            'state': str(node.state),
            # 'value': value,
            'depth': node.depth,
        }
        logger.info(f'Node index #{node.tree_index}: {cls.nodes[node.tree_index]}')

    @classmethod
    def save_edge(cls, parent: Node, child: Node) -> None:
        if not cls.tree_wanted:
            return
        if not hasattr(parent, 'tree_index'):
            parent.tree_index = cls.node_counter
            cls.node_counter += 1
        if not hasattr(child, 'tree_index'):
            child.tree_index = cls.node_counter
            cls.node_counter += 1
        cls.edges.append((parent.tree_index, child.tree_index))


class Stats:
    def __init__(self, prior: float):
        self.N = 0
        self.W = .0
        self.P = prior

    def update(self, value: float):
        self.N += 1
        self.W += value
        logger.debug(f'Updating edge with value {value}: {self}')

    @property
    def Q(self) -> float:  # noqa
        if self.N == 0:
            return 0
        return self.W / self.N

    def dict(self) -> dict:
        return {
            'N': self.N,
            'W': self.W,
            'Q': self.Q,
            'P': self.P,
        }

    def __repr__(self) -> str:
        text = f'N={self.N}, W={self.W:.3f}, Q={self.Q:.3f}'
        if self.P is not None:
            text += f', P={self.P:.3f}'
        return text


class Edge:
    def __init__(
        self,
        in_node: Node,
        out_node: Node,
        action: int,
        prior: float = None,
    ):
        self.in_node = in_node
        self.out_node = out_node
        self.action = action
        self.stats = Stats(prior)

    @property
    def player_turn(self) -> int:
        return self.in_node.player_turn

    def __repr__(self) -> str:
        return f"{self.action}, {self.in_node}->{self.out_node}"


def log_trajectory(bread: list[Edge]) -> None:
    if not bread:
        return
    text = 'Leaf trajectory:'
    for edge in bread:
        text += f'\n{edge}'
    logger.debug(text)


def get_nested_nodes(s):
    if not hasattr(s, 'children'):
        return s.tree_index
    return {
        s.tree_index: [get_nested_nodes(n) for n in s.children]
    }
