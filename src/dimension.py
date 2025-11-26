import networkx as nx
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.serif'] = ['Times New Roman']  # Especifica Times New Roman
plt.rc('font', size=18)  # Ajusta el tamaño de la fuente según prefieras 18
plt.rcParams['grid.linestyle'] = '--'  # Estilo de línea del grid
plt.rcParams['grid.linewidth'] = 0.4   # Ancho de línea del grid
plt.rcParams['grid.alpha'] = 0.6       # Transparencia del grid
plt.rcParams['axes.grid'] = True       # Activa el grid de manera predeterminada en todos los ejes

def nx_BFS(G, start_node):
    #We will be using the BFS algorithm
    queue = deque()
    distance = {node: np.inf for node in G}
    
    #Initialize the algorithm with the first node of G
    queue.append(start_node)
    distance[start_node] = 0 
    
    
    #Loop through the neighbors and add any unvisited ones to the queue.
    while len(queue) > 0:
        current_node = queue.popleft()
        for neighbor in G.neighbors(current_node):
            if distance[neighbor] == np.inf:
                queue.append(neighbor)
                distance[neighbor] = distance[current_node] + 1

    return distance

def mu_L(G, current_node, L):
    distance = nx_BFS(G, current_node)
    sum = 0
    for dist in distance.values():
        if dist > 0 and dist <= L:
            sum += 1
    mu = sum/(len(G)-1)
    return mu

def dimension_analysis(G, diameter, q):
    all_nodes = list(G.nodes)
    sample_size = int(len(G)/10) #This is the number of nodes for doing the average, we will be choosing random nodes
    random_nodes = random.sample(all_nodes , sample_size)
    dimension = []   
    if q == 1:
        for L in range (1, diameter + 1 ):
            mu_value = 0.
            for node in random_nodes:
                mu_value += mu_L(G, node, L)
            mu_value /= sample_size
            dimension.append(mu_value)
    else:
        for L in range (1, diameter + 1 ):
            mu_value = 0.
            for node in random_nodes:
                mu_value += mu_L(G, node, L)**(q-1)
            mu_value /= sample_size
            dimension.append(mu_value)
            
    return dimension

def fit_dimension(G, diameter, q):
    dimension = dimension_analysis(G,diameter, q)
    L_values = list(range(1, diameter + 1))
    log_L = np.log(L_values)
    log_mu = np.log(dimension)
    
    #Choose the values for the fit
    log_L_fit = log_L[:]
    log_mu_fit = log_mu[:]
    
    slope, intercept = np.polyfit(log_L_fit, log_mu_fit, 1)
    
    if q == 1:
        dimension_value = slope
    else:
        dimension_value = slope/(q-1)
    
    # Plot
    plt.figure(figsize=(6,4))
    plt.scatter(log_L, log_mu, label='Data')
    plt.plot(log_L_fit, slope * log_L_fit + intercept, color='red', label='Fit')
    plt.xlabel('ln L')
    if q == 1:
        plt.ylabel(r'Average $\ln \mu_L$')
        plt.title('Information Dimension (D1) Estimate')
    else:
        plt.ylabel(r'$\ln \langle \mu_L^{q-1} \rangle_X$')
        plt.title(f'Dimension Estimate (q={q})')
    # plt.legend()
    plt.tight_layout()
    plt.show()  # Displays the plot
    
    return dimension_value