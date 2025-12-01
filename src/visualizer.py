import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def hc_scale_free_graph_generator(n, m, a, seed=None):

    if n < m:
        raise ValueError('n debe ser >= m')
    if seed is not None:
        random.seed(seed)

    G = nx.Graph() 

    for x in range(m):
        G.add_node(x)
        for y in range(x + 1, m):
            G.add_edge(x, y)
    
    active_nodes = list(range(m))
    k = {x: m - 1 for x in range(m)}

    yield G.copy(), list(active_nodes), None

    i = m
    while i < n:
        G.add_node(i)
        k[i] = 0
        
        for j in active_nodes:
            G.add_edge(i, j)
            k[j] += 1
            k[i] += 1
        
        active_nodes.append(i)
        
        weights = [1 / (a + k[j]) for j in active_nodes]
        total_weight = sum(weights)
        
        r = random.random() * total_weight
        cumulative = 0.0
        remove_idx = -1
        
        for idx, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                remove_idx = idx
                break
        
        if remove_idx != -1:
            active_nodes.pop(remove_idx)

        yield G.copy(), list(active_nodes), i
        i += 1

def generate_interactive_video(n=50, m=3, a=0.5, seed=42):

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.close(fig)

    steps = list(hc_scale_free_graph_generator(n, m, a, seed))
    
    final_G = steps[-1][0]
    pos = nx.spring_layout(final_G, seed=seed, iterations=150, k=2/np.sqrt(n))

    def update(frame_idx):
        ax.clear()
        current_G, current_active, new_node = steps[frame_idx]
        
        node_colors = []
        node_sizes = []
        
        for node in current_G.nodes():
            size = 120 if node in current_active else 60
            node_sizes.append(size)

            if node in current_active:
                node_colors.append('#ff4d4d') # Actives
            elif node == new_node:
                node_colors.append('#ffa500') # New
            else:
                node_colors.append('#1f78b4') # Inactives
        
        current_pos = {node: pos[node] for node in current_G.nodes()}
        
        nx.draw(current_G, current_pos, ax=ax, node_color=node_colors, 
                node_size=node_sizes, edge_color="#b3b3b3", alpha=0.7, width=0.8, with_labels=False)
        
        ax.set_title(f"Step {frame_idx}/{n-m} | N={len(current_G)} | Actives: {len(current_active)} (Red)", fontsize=10)
        ax.axis('off')

    ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=200, repeat=False)
    return ani