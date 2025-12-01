import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

def get_cumulative_mu(G, start_node, max_L):
    """
    Efficiently calculates the cumulative distribution of nodes (mu_L)
    at distance L from start_node using a single BFS.
    """
    N = len(G)
    
    # Optimized BFS: Get all shortest path lengths from the start_node
    # Returns a dict {target_node: distance}
    path_lengths = nx.single_source_shortest_path_length(G, start_node)
    
    # Extract distances (excluding distance 0 to itself)
    distances = np.array([dist for dist in path_lengths.values() if dist > 0])
    
    if len(distances) == 0:
        return np.zeros(max_L + 1)

    # Create a histogram: counts[d] = number of nodes at exact distance d
    counts = np.bincount(distances)
    
    # Cumulative sum: number of nodes at distance <= L
    cumulative = np.cumsum(counts)
    
    # Normalize by (N-1) to get the fraction mu_L
    mu_values = cumulative / (N - 1)
    
    # Handle array size: pad with the last value up to max_L
    # (Because if the node sees the whole network at dist 5, dist 6 is also 1.0)
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
                 If q=1, returns <ln(mu)>.
                 If q!=1, returns <mu^(q-1)>.
    """
    nodes = list(G.nodes())
    sample_size = max(1, int(len(nodes) * sample_ratio))
    sampled_nodes = random.sample(nodes, sample_size)
    
    # Use a safe approximation for diameter to size the arrays
    # (calculating exact diameter is too slow for large networks)
    try:
        diameter = nx.approximation.diameter(G)
    except:
        diameter = len(G) // 2
        
    sum_moments = np.zeros(diameter + 1)
    valid_samples = 0
    
    for node in sampled_nodes:
        # Get the mu_L curve for this node
        mu_array = get_cumulative_mu(G, node, diameter)
        
        # Avoid log(0) or division by zero with a tiny epsilon
        mu_array = np.maximum(mu_array, 1e-12)
        
        if q == 1:
            # Information Dimension: Average of logarithms <ln(mu)>
            moment = np.log(mu_array)
        else:
            # Correlation/Capacity: Average of powers <mu^(q-1)>
            moment = mu_array ** (q - 1)
            
        sum_moments += moment
        valid_samples += 1
        
    avg_moments = sum_moments / valid_samples
    
    # Return L from 1 to diameter (index 0 is distance 0, which we ignore)
    return np.arange(1, diameter + 1), avg_moments[1:]

def fit_dimension(L, moments, q, min_L=None, max_L=None, plot=False):
    """
    Performs the linear regression to find D_q.
    
    Args:
        L: Array of distances.
        moments: The data returned by calculate_dimension_data.
        q: The dimension order.
        plot: If True, shows a simple plot of the fit.
        min_L: Start distance for the fit (filters out small scale effects).
        max_L: End distance for the fit (filters out finite size saturation).
        
    Returns:
        D_val: The estimated dimension.
    """
    # Prepare data for log-log fit
    x_data = np.log(L)
    
    if q == 1:
        # For D1, we already have <ln(mu)>. The equation is linear in log-log.
        # Slope is directly D1.
        y_data = moments
        ylabel = r'$\langle \ln \mu_L \rangle$'
    else:
        # For Dq (q!=1), we have <mu^(q-1)>. We need ln(<mu^(q-1)>).
        # Slope is Dq * (q-1).
        y_data = np.log(moments)
        ylabel = r'$\ln \langle \mu_L^{q-1} \rangle$'
    
    if min_L is None: min_L = 1
    if max_L is None: max_L = L[-1]
    
    mask = (L >= min_L) & (L <= max_L)
    x_fit = x_data[mask]
    y_fit = y_data[mask]
    
    if len(x_fit) < 2:
        print("Warning: Not enough points for fit in the selected range.")
        return 0.0
        
    # Linear Regression (y = mx + c)
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    
    # Calculate D based on q
    if q == 1:
        D_val = slope
    else:
        D_val = slope / (q - 1)
        
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(x_data, y_data, s=15, label='Data', color='maroon')
        plt.plot(x_data, slope * x_data + intercept, color='dodgerblue', label=fr'Fit ($D={D_val:.3f}$)')
        plt.axvline(x=np.log(min_L), color='dimgray', linestyle=':', alpha=0.5)
        plt.axvline(x=np.log(max_L), color='dimgray', linestyle=':', alpha=0.5)
        plt.xlabel(r'$\ln L$')
        plt.ylabel(ylabel)
        plt.title(f'Dimension $q={q}$')
        plt.legend()
        plt.show()
        
    return D_val
