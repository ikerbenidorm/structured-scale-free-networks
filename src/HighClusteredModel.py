import networkx as nx
import random

def hc_scale_free_graph(n, m=3, a=0.5, directed=True, seed=None):
    """
    Generates a Highly Clustered Scale-Free Network using the Klemm-Eguíluz model.

    Reference:
    Klemm, K., & Eguíluz, V. M. Highly clustered scale-free networks. 
    Phys. Rev. E 65, 036123 (2002).

    Parameters:
    -----------
    n : int
        Final number of nodes in the network.
    m : int
        Number of initially active nodes (and edges added per step).
    a : float
        Parameter 'a' controlling the deactivation probability. 
        See Eq. (1) in the reference paper.
    directed : bool
        If True, returns a DiGraph (directed). If False, returns a Graph.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    G : networkx.Graph or networkx.DiGraph
        The generated network.
    """
    if n < m:
        raise ValueError('The total number of nodes (n) must be >= initial nodes (m).')

    # Set the seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Initialize Graph
    G = nx.DiGraph() if directed else nx.Graph()

    # 1. Initialization: Fully connected graph of m active nodes [cite: 101]
    for x in range(m):
        for y in range(x + 1, m):
            G.add_edge(x, y)
            if directed:
                G.add_edge(y, x)

    active_nodes = list(range(m))     # List of active nodes [cite: 98]
    # Initial degrees: in a fully connected size m, everyone has m-1 links
    k = {x: m - 1 for x in range(m)}  

    # 2. Growth and Deactivation Loop
    i = m
    while i < n:
        # --- Growth Step [cite: 102-104] ---
        G.add_node(i)
        k[i] = 0

        # Connect new node i to all currently active nodes
        for j in active_nodes:
            # Direction: From new node i -> active node j (incoming for j)
            G.add_edge(i, j) 
            k[j] += 1

            # If undirected, we must also count the degree increase for i
            if not directed:
                k[i] += 1

        active_nodes.append(i) # [cite: 105]

        # --- Deactivation Step [cite: 105-107] ---
        # Probability P(kj) ~ 1 / (a + kj)
        weights = [1 / (a + k[j]) for j in active_nodes]
        total_weight = sum(weights)

        # Select one node to deactivate based on weights
        r = random.random() * total_weight
        cumulative = 0.0
        for idx, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                # Deactivate this node (remove from active list)
                active_nodes.pop(idx)
                break

        i += 1

    return G
