import networkx as nx
import random
import math
import torch 
import numpy as np
def generate_dag(L, N, C, seed=None):
    
    """
    L = number of layers in the DAG
    N = total number of nodes in the DAG
    C = maximum number of children per node
    
    """
    if seed is not None:
        random.seed(seed)

    G = nx.DiGraph()
    
    # Step 1: Partition nodes into L layers
    layers = [[] for _ in range(L)]
    
    # Assign at least one node per layer to ensure full depth
    remaining_nodes = N - L
    for i in range(L):
        layers[i].append(i)  # assign one node per layer initially
    
    node_idx = L
    while remaining_nodes > 0:
        layer = random.randint(0, L - 1)
        layers[layer].append(node_idx)
        node_idx += 1
        remaining_nodes -= 1
    
    # Step 2: Add nodes to the graph
    for idx, layer_nodes in enumerate(layers):
        G.add_nodes_from(layer_nodes, layer=idx)

    # Step 3: Connect nodes respecting DAG structure (only forward edges)
    for i in range(L - 1):
        current_layer = layers[i]
        next_layers = sum(layers[i+1:], [])  # all nodes in subsequent layers
        
        for parent in current_layer:
            num_children = random.randint(1, min(C, len(next_layers)))
            children = random.sample(next_layers, num_children)
            for child in children:
                G.add_edge(parent, child)

    # Step 4: Assign unique binary strings of length ceil(log2(N))
    binary_len = math.ceil(math.log2(N))
    node_ids = list(G.nodes())
    random.shuffle(node_ids)  # randomize assignment order
    binaries = [format(i, f'0{binary_len}b') for i in range(N)]
    node_binary_map = dict(zip(node_ids, binaries))
    
    nx.set_node_attributes(G, node_binary_map, 'binary_id')
    
    return G


import networkx as nx
import matplotlib.pyplot as plt
def visualize_dag_layers(G, highlight_nodes=None):
    if highlight_nodes is None:
        highlight_nodes = []

    layers = {}
    for node, data in G.nodes(data=True):
        layer = data['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)
    
    pos = {}
    max_width = max(len(nodes) for nodes in layers.values())

    for layer_idx, nodes in layers.items():
        width = len(nodes)
        spacing = max_width / (width + 1)
        y = -layer_idx  # negative so root is on top
        for i, node in enumerate(sorted(nodes)):
            x = (i + 1) * spacing
            pos[node] = (x, y)

    # Node coloring
    node_colors = []
    for node in G.nodes():
        if node in highlight_nodes:
            node_colors.append('orange')
        else:
            node_colors.append('lightblue')

    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos,
        with_labels=True,
        arrows=True,
        arrowsize=20,
        node_size=1200,
        node_color=node_colors,
        font_size=10
    )

    labels = nx.get_node_attributes(G, 'binary_id')
    nx.draw_networkx_labels(
        G,
        {k: (v[0], v[1] - 0.1) for k, v in pos.items()},
        labels=labels,
        font_size=8,
        font_color='darkblue'
    )

    plt.title("Hierarchical DAG Visualization")
    plt.axis('off')
    plt.show()



def compute_all_subtrees(G):
    subtrees = {}

    def dfs(node, visited):
        if node in subtrees:  # memoization
            return subtrees[node]

        descendants = set()
        for child in G.successors(node):
            if child not in visited:
                visited.add(child)
                descendants.add(child)
                descendants |= dfs(child, visited)
        subtrees[node] = descendants
        return descendants

    for node in G.nodes():
        dfs(node, visited=set())

    return subtrees



from collections import deque

def bfs_subtree_ordering(G, root):
    """
    Perform BFS over the subtree rooted at `root` in a DAG G.

    Args:
        G (networkx.DiGraph): The DAG.
        root (node): The starting node.

    Returns:
        List[node]: Nodes in BFS order starting from `root`.
    """
    visited = set()
    queue = deque([root])
    bfs_order = []

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        bfs_order.append(node)
        successors = list(G.successors(node))
        # randomly shuffle successors to introduce variability in BFS order
        random.shuffle(successors)
        for neighbor in successors:
            if neighbor not in visited:
                queue.append(neighbor)
    
    return bfs_order[1:]


def dfs_subtree_ordering(G, root):
    """
    Perform DFS over the subtree rooted at `root` in a DAG G.

    Args:
        G (networkx.DiGraph): The DAG.
        root (node): The starting node.

    Returns:
        List[node]: Nodes in DFS order starting from `root`.
    """
    visited = set()
    dfs_order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        # skip appending root here if you want to match bfs_order[1:] pattern
        if node != root:
            dfs_order.append(node)
        successors = list(G.successors(node))
        random.shuffle(successors)  # optional variability in DFS traversal
        for neighbor in successors:
            dfs(neighbor)

    dfs(root)
    return dfs_order


def get_dag_batch(dag, block_size = 100, batch_size = 64, device_type = "cuda", device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    nodes = list(dag.nodes())
    text = []
    while len(text) < (block_size+1) * batch_size:
        node = random.choice(nodes)  # randomly select a node to start the batch
        dfs_order = dfs_subtree_ordering(dag, node) + [node]  # include the root node at the end
        
        #bfs_order = bfs_subtree_ordering(dag, node) + [node] # root node is the 
        text.extend(dfs_order)
    text = np.array(text)
        
    # create a batch of nodes in BFS order
    ix = torch.randint(len(text)- block_size, (batch_size,))
    
    x = torch.stack([torch.from_numpy((text[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((text[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y