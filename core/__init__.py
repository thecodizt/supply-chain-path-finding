from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import random


def dfs_topological_sort_with_sources_sinks(edge_list):
    graph = defaultdict(list)

    for source, target in edge_list:
        graph[source].append(target)

    visited = set()
    stack = []
    in_degree = defaultdict(int)

    # Calculate in-degrees first
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)

    # Perform DFS
    all_nodes = list(graph.keys())  # Create a list of all nodes
    for node in all_nodes:
        if node not in visited:
            dfs(node)

    sorted_order = stack[::-1]

    # Identify source and sink nodes
    all_nodes = set(graph.keys()) | set(
        node for neighbors in graph.values() for node in neighbors
    )
    source_nodes = [node for node in all_nodes if in_degree[node] == 0]
    sink_nodes = [node for node in all_nodes if not graph[node]]

    return sorted_order, source_nodes, sink_nodes


def handle_padding(matrix):
    # Augumenting super source and sink
    num_nodes = len(matrix)
    layer = [0 for i in range(num_nodes)]

    matrix = [layer] + matrix + [layer]

    for i in range(num_nodes + 2):
        matrix[i] = [0] + matrix[i] + [0]

    return matrix


def convert_list_to_matrix(items_list, size, index_offset=0):
    matrix = [[0 for i in range(size)] for j in range(size)]

    for item in items_list:
        if len(item) == 2:
            matrix[item[0] - index_offset][item[1] - index_offset] = 1
        else:
            matrix[item[0] - index_offset][item[1] - index_offset] = item[2]

    return matrix


def calculate_max_flow(paths, capacity_matrix, super_source, super_target):
    G = nx.DiGraph()
    for path in paths:
        for edge in path:
            G.add_edge(*edge, capacity=capacity_matrix[edge[0]][edge[1]])
    return nx.maximum_flow_value(G, super_source, super_target)


def visualize_network_with_paths(capacity_matrix, cost_matrix, paths, max_flow):
    G = nx.DiGraph()
    for i in range(len(capacity_matrix)):
        for j in range(len(capacity_matrix)):
            if capacity_matrix[i][j] > 0:
                G.add_edge(i, j, capacity=capacity_matrix[i][j])

    pos = nx.circular_layout(G)

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=12,
        arrows=True,
    )

    combined_paths = sum(paths, [])
    # remove duplicate edges
    combined_paths = list(dict.fromkeys(combined_paths))

    combined_cost = sum(cost_matrix[edge[0]][edge[1]] for edge in combined_paths)

    nx.draw_networkx_edges(
        G, pos, edgelist=combined_paths, edge_color="g", width=2, ax=ax
    )

    edge_labels = {(i, j): f"{capacity_matrix[i][j]}" for (i, j) in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    ax.set_title(
        f"Network with highlighted routes\nTotal Cost: {combined_cost}, Max Flow: {max_flow}"
    )

    return fig


def combinative_greedy_dfs(
    cost_matrix,
    capacity_matrix,
    super_source,
    super_target,
    target_demand,
    nodes_to_avoid=None,
):
    if nodes_to_avoid is None:
        nodes_to_avoid = []

    def dfs(node, path, cost):
        if node == super_target:
            paths.append((path, cost))
            return

        options = []
        for neighbor in range(len(capacity_matrix)):
            if capacity_matrix[node][neighbor] > 0 and neighbor not in nodes_to_avoid:
                options.append(neighbor)

        costs = [(option, cost_matrix[node][option]) for option in options]
        costs.sort(key=lambda x: x[1])

        for option, _ in costs:
            dfs(option, path + [(node, option)], cost + cost_matrix[node][option])

    paths = []
    dfs(super_source, [], 0)
    paths.sort(key=lambda x: x[1])  # Sort paths by cost

    for r in range(1, len(paths) + 1):
        for combo in combinations(paths, r):
            combined_paths = [path for path, _ in combo]
            combined_cost = sum(
                cost_matrix[edge[0]][edge[1]]
                for path in combined_paths
                for edge in path
            )
            max_flow = calculate_max_flow(
                paths=combined_paths,
                capacity_matrix=capacity_matrix,
                super_source=0,
                super_target=len(capacity_matrix) - 1,
            )

            if max_flow >= target_demand:
                return [(combined_paths, combined_cost, max_flow)]

    return []  # If no solution is found


def check_if_solution_is_possible(
    cost_matrix, capacity_matrix, super_source, super_target, target_demand
):
    edges = []

    for i in range(len(capacity_matrix)):
        for j in range(len(capacity_matrix)):
            if capacity_matrix[i][j] > 0:
                edges.append((i, j))

    max_flow = calculate_max_flow(
        paths=[edges],
        capacity_matrix=capacity_matrix,
        super_source=super_source,
        super_target=super_target,
    )

    return max_flow >= target_demand


def handle_transforms(initial_edge_list, capacity_list, cost_list, num_nodes):
    initial_matrix = convert_list_to_matrix(
        initial_edge_list, num_nodes, index_offset=1
    )
    initial_matrix = handle_padding(initial_matrix)

    capacity_matrix = convert_list_to_matrix(capacity_list, num_nodes, index_offset=1)
    capacity_matrix = handle_padding(capacity_matrix)

    cost_matrix = convert_list_to_matrix(cost_list, num_nodes, index_offset=1)
    cost_matrix = handle_padding(cost_matrix)

    mesh = []

    for i in range(len(capacity_matrix)):
        for j in range(len(capacity_matrix)):
            if capacity_matrix[i][j] > 0:
                mesh.append((i, j))

    sorted_order, source_nodes, sink_nodes = dfs_topological_sort_with_sources_sinks(
        edge_list=mesh
    )

    for source in source_nodes:
        initial_matrix[0][source] = 1
        capacity_matrix[0][source] = np.inf
        cost_matrix[0][source] = 0.00001

    for sink in sink_nodes:
        initial_matrix[sink][len(initial_matrix) - 1] = 1
        capacity_matrix[sink][len(capacity_matrix) - 1] = np.inf
        cost_matrix[sink][len(cost_matrix) - 1] = 0.00001

    for edge in initial_edge_list:
        cost_matrix[edge[0]][edge[1]] = 0

    initial_sources = [x[0] for x in initial_edge_list]
    initial_targets = [x[1] for x in initial_edge_list]

    for i in initial_sources:
        if i in source_nodes:
            cost_matrix[0][i] = 0

    for i in initial_targets:
        if i in sink_nodes:
            cost_matrix[i][len(cost_matrix) - 1] = 0

    return initial_edge_list, capacity_matrix, cost_matrix


def generate_random_graph(num_nodes, max_capacity=20, max_cost=20):
    # Generate capacity list
    capacity_list = []
    for i in range(1, num_nodes + 1):
        for j in range(i + 1, num_nodes + 1):
            if random.random() < 0.7:  # 70% chance of edge existing
                capacity = random.randint(1, max_capacity)
                capacity_list.append((i, j, capacity))

    # Generate cost list (same structure as capacity list)
    cost_list = [
        (edge[0], edge[1], random.randint(1, max_cost)) for edge in capacity_list
    ]

    # Generate initial edge list (subset of capacity list)
    initial_edge_list = []
    for edge in capacity_list:
        if random.random() < 0.3:  # 30% chance of being in initial list
            initial_edge_list.append((edge[0], edge[1]))

    return initial_edge_list, capacity_list, cost_list
