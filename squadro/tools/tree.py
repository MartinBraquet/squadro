import json
from pathlib import Path

import plotly.graph_objects as go
from igraph import Graph


def get_nested_nodes(s):
    if not hasattr(s, 'children'):
        return s.tree_index
    return {
        s.tree_index: [get_nested_nodes(n) for n in s.children]
    }


def plot_tree(edge_labels=True):
    tree_directory = Path('results')

    edges = json.load(open(tree_directory / 'edges.json'))
    nodes = json.load(open(tree_directory / 'nodes.json'))
    nodes = {int(k): v for k, v in nodes.items()}
    # labels = list(range(max(max(e) for e in edges) + 1))
    nested_nodes = json.load(open(tree_directory / 'nested_nodes.json'))
    if edge_labels:
        edge_labels = json.load(open(tree_directory / 'edge_values.json'))
        edge_labels = {eval(key): value for key, value in edge_labels.items()}

    def get_n(l: list):
        assert isinstance(l, list)
        n_ = len(l)
        for i, e in enumerate(l):
            if isinstance(e, dict):
                assert len(e) == 1
                n_ = max(n_, get_n(list(e.values())[0]))
        return n_

    # n = len(eval(nodes[0]["state"])[0])
    n = get_n([nested_nodes])

    n_vertices = max(max(e) for e in edges) + 1

    G = Graph(edges=edges)

    pos = {}
    Y = max(n['depth'] for n in nodes.values())

    def walk_nodes(l, prev_x=0, y=0):
        assert isinstance(l, list)
        x = prev_x
        for i, e in enumerate(l):
            if y != 0:
                x = prev_x + (i - n // 2) * n ** (Y - y)
            if isinstance(e, dict):
                assert len(e) == 1
                k = list(e.keys())[0]
                pos[int(k)] = (x, Y - y)
                walk_nodes(list(e.values())[0], x, y + 1)
            else:
                assert isinstance(e, int)
                pos[int(e)] = (x, Y - y)

    walk_nodes([nested_nodes])


    # pos = {i: (x, D - y) for i, (x, y) in node_ordering.items()}

    E = [e.tuple for e in G.es]  # list of edges

    L = len(pos)
    Xn = [pos[k][0] for k in range(L)]
    Yn = [pos[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe += [pos[edge[0]][0], pos[edge[1]][0], None]
        Ye += [pos[edge[0]][1], pos[edge[1]][1], None]

    if nodes[0].get('value') is not None:
        labels = [nodes[k]['value'] for k in pos.keys()]
    else:
        labels = list(map(str, range(n_vertices)))

    if len(labels) != L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=labels[k],
                x=pos[k][0], y=pos[k][1],
                xref='x1', yref='y1',
                font=dict(color='rgb(250,250,250)', size=10),
                showarrow=False)
        )

    if edge_labels:
        for edge, label in edge_labels.items():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            annotations.append(
                dict(
                    x=mid_x,
                    y=mid_y,
                    text=label,
                    showarrow=False,
                    font=dict(color="red", size=12)
                )
            )

    fig = go.Figure()
    # fig.update_layout(
    #     width=1600,  # Set figure width to 1000 pixels
    #     height=700,  # Set figure height to 800 pixels
    # )

    fig.add_trace(go.Scatter(x=Xe,
                             y=Ye,
                             mode='lines',
                             line=dict(color='rgb(210,210,210)', width=1),
                             hoverinfo='none'
                             ))
    fig.add_trace(go.Scatter(x=Xn,
                             y=Yn,
                             mode='markers',
                             name='bla',
                             marker=dict(symbol='circle-dot',
                                         size=25,
                                         color='#6175c1',  # '#DB4551',
                                         line=dict(color='rgb(50,50,50)', width=1)
                                         ),
                             opacity=0.8
                             ))
    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )

    fig.update_layout(title='Minimax Tree',
                      annotations=annotations,
                      font_size=12,
                      showlegend=False,
                      xaxis=axis,
                      yaxis=axis,
                      margin=dict(l=40, r=40, b=50, t=50),
                      hovermode='closest',
                      plot_bgcolor='rgb(248,248,248)'
                      )
    fig.show()
