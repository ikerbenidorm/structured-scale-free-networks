import networkx as nx


def build_graph_from_edges(edge_list):
# Construct a graph from a list of edges without using nx.Graph()
    graph_dict = {}
    nodes = set()      #This creates an EMPTY "set"; in this type of data, each element is unique.

    for edge in edge_list:
        # Skip headings if they exist in the file
        if edge.startswith('#') or not edge.strip():
            continue

        # Divide the line into nodes
        parts = edge.strip().split()
        if len(parts) >= 2:
            node1, node2 = parts[0], parts[1]
            nodes.add(node1)
            nodes.add(node2)

            # Add node1 -> node2 connection
            if node1 not in graph_dict:
                graph_dict[node1] = []
            graph_dict[node1].append(node2)

            # Add node2 -> node1 connection (undirected graph)
            if node2 not in graph_dict:
                graph_dict[node2] = []
            graph_dict[node2].append(node1)

    return graph_dict, nodes

def calculate_degree_distribution_manual(edge_list): # from a list of nodes
#Calculate the degree distribution without networkx
    graph_dict, nodes = build_graph_from_edges(edge_list)

    # First we obtain all the degrees
    degrees = [len(graph_dict.get(node, [])) for node in nodes]

    # Then we count the frequencies
    unique_degrees = set(degrees)
    degree_count = {degree: degrees.count(degree) for degree in unique_degrees}

    # Calculate normalized distribution
    total_nodes = len(nodes)
    degree_distribution = {degree: count / total_nodes for degree, count in degree_count.items()}

    # Create a minimal graph to be read by NetworkX in other analyses
    G = MinimalGraph(graph_dict)

    return degree_distribution, degree_count, graph_dict, G

def read_edgelist_file(filename):
#Read an .edgelist file and return the list of edges
    try:
        with open(filename, 'r') as file:
            edges = file.readlines()
        return edges
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo '{filename}'")
        return None
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None
    
def calculate_degree_distribution_Graph(nx_graph): # from a NetworkX Graph
#Calculate the degree distribution without networkx
    graph_dict = {}
    for node in nx_graph.nodes():
      # Get all neighbors of the node
      graph_dict[node] = list(nx_graph.neighbors(node))
    nodes = list(nx_graph.nodes())

    # First we obtain all the degrees
    degrees = [len(graph_dict.get(node, [])) for node in nodes]

    # Then we count the frequencies
    unique_degrees = set(degrees)
    degree_count = {degree: degrees.count(degree) for degree in unique_degrees}

    # Calculate normalized distribution
    total_nodes = len(nodes)
    degree_distribution = {degree: count / total_nodes for degree, count in degree_count.items()}

    # Graph to be read by NetworkX in other analyses
    G = nx_graph

    return degree_distribution, degree_count, graph_dict, G