import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

def get_cumulative_mu(G, start_node, max_L):
    """
    Calculates the cumulative distribution of nodes
    at distance L from start_node using a single BFS.
    """
    N = len(G)
    
    path_lengths = nx.single_source_shortest_path_length(G, start_node)
    
    distances = np.array([dist for dist in path_lengths.values() if dist > 0])
    
    if len(distances) == 0:
        return np.zeros(max_L + 1)

    counts = np.bincount(distances)
    
    cumulative = np.cumsum(counts)
    
    mu_values = cumulative / (N - 1)
    
    result = np.ones(max_L + 1) * mu_values[-1]
    limit = min(len(mu_values), max_L + 1)
    result[:limit] = mu_values[:limit]
    
    return result

def calculate_dimension_data(G, q, sample_ratio=0.1):
    """
    Calculates the averaged statistical moments required for the dimension fit.
    
    Returns:
        L_values: Array [1, 2, ... diameter]
        moments: The averaged Y-axis values.
    """
    nodes = list(G.nodes())
    sample_size = max(1, int(len(nodes) * sample_ratio))
    sampled_nodes = random.sample(nodes, sample_size)
    
    try:
        diameter = nx.approximation.diameter(G)
    except:
        diameter = len(G) // 2
        
    sum_moments = np.zeros(diameter + 1)
    valid_samples = 0
    
    for node in sampled_nodes:
        mu_array = get_cumulative_mu(G, node, diameter)
        
        mu_array = np.maximum(mu_array, 1e-12) # Avoid log(0) or division by zero with an epsilon
        
        if q == 1:
            moment = np.log(mu_array)
        else:
            moment = mu_array ** (q - 1)
            
        sum_moments += moment
        valid_samples += 1
        
    avg_moments = sum_moments / valid_samples

    return np.arange(1, diameter + 1), avg_moments[1:]

def fit_dimension(L, moments, q, min_L=None, max_L=None, plot=False):
    """
    Linear regression to find D_q.
    
    Args:
        L: Array of distances.
        moments: The data returned by calculate_dimension_data.
        q: The dimension order.
        plot: If True, shows a plot of the fit.
        min_L: Start distance for the fit.
        max_L: End distance for the fit.
        
    Returns:
        D_val: The estimated dimension.
    """
    x_data = np.log(L)
    
    if q == 1:
        y_data = moments
        ylabel = r'$\langle \ln \mu_L \rangle$'
    else:
        y_data = np.log(moments)
        ylabel = r'$\ln \langle \mu_L^{q-1} \rangle$'
    
    if min_L is None: min_L = 1
    if max_L is None: max_L = L[-1]
    
    mask = (L >= min_L) & (L <= max_L)
    x_fit = x_data[mask]
    y_fit = y_data[mask]
    
    if len(x_fit) < 2:
        print("Error: Not enough points for fit in the selected range.")
        return 0.0
        
    slope, b = np.polyfit(x_fit, y_fit, 1)
    
    if q == 1:
        D_val = slope
    else:
        D_val = slope / (q - 1)
        
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(x_data, y_data, s=15, label='Data', color='maroon')
        plt.plot(x_data, slope * x_data + b, color='dodgerblue', label=fr'Fit ($D={D_val:.3f}$)')
        plt.axvline(x=np.log(min_L), color='dimgray', linestyle=':', alpha=0.5)
        plt.axvline(x=np.log(max_L), color='dimgray', linestyle=':', alpha=0.5)
        plt.xlabel(r'$\ln L$')
        plt.ylabel(ylabel)
        plt.title(f'Dimension $q={q}$')
        plt.legend()
        plt.show()
        
    return D_val
