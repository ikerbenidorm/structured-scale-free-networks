import matplotlib
# matplotlib.use('Agg')  # Backend sin GUI, ideal para generar imágenes
import matplotlib.pyplot as plt
import networkx as nx

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.serif'] = ['Times New Roman']  # Especifica Times New Roman
plt.rc('font', size=18)  # Ajusta el tamaño de la fuente según prefieras 18
plt.rcParams['grid.linestyle'] = '--'  # Estilo de línea del grid
plt.rcParams['grid.linewidth'] = 0.4   # Ancho de línea del grid
plt.rcParams['grid.alpha'] = 0.6       # Transparencia del grid
plt.rcParams['axes.grid'] = True       # Activa el grid de manera predeterminada en todos los ejes

def plot_network_simple(G, filename, l, clustering=None, config_name="", dpi = 600):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=20, alpha=0.7, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    plt.title(f'{config_name}\n{l} nodes | Clustering: {clustering:.3f}')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', dpi=dpi)
    plt.close()