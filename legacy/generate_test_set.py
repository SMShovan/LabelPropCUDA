import networkx as nx
import numpy as np
import argparse
import os

def generate_erdos_renyi_graph(n_nodes, probability, s_rand, e_rand, output_path):
    """
    Generates an Erdős–Rényi graph with n_nodes, probability of edge creation,
    and random edge weights between sRand and eRand.
    Saves the adjacency matrix to the specified output path.
    """
    # Generate the graph
    graph = nx.erdos_renyi_graph(n=n_nodes, p=probability)

    # Ensure the graph is connected
    while not nx.is_connected(graph):
        graph = nx.erdos_renyi_graph(n=n_nodes, p=probability)

    # Add random weights to the edges
    for u, v in graph.edges():
        graph[u][v]['weight'] = np.random.randint(s_rand, e_rand + 1)

    # Convert to adjacency matrix with weights
    adjacency_matrix = nx.to_numpy_array(graph, weight='weight', dtype=int)

    # Save the adjacency matrix to the output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, adjacency_matrix, fmt='%d')

    print(f"Erdős–Rényi graph with {n_nodes} nodes saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an Erdős–Rényi graph and save its adjacency matrix.")
    parser.add_argument("--probability", type=float, help="Edge probability for the Erdős–Rényi graph.")
    parser.add_argument("--nodes", type=int, help="Number of nodes in the graph.")
    parser.add_argument("--sRand", type=int, help="Minimum edge weight.")
    parser.add_argument("--eRand", type=int, help="Maximum edge weight.")
    parser.add_argument("--output", type=str, default="/mnt/stor/ceph/csc/sdas-lab/Shovan/adjacency_matrix.txt", help="Output file path.")

    args = parser.parse_args()

    generate_erdos_renyi_graph(args.nodes, args.probability, args.sRand, args.eRand, args.output)


#python generate_test_set.py --probability 0.7 --nodes 1000000 --eR--sRand 1 and 10 --output /mnt/stor/ceph/csc/sdas-lab/Shovan/adjacency_matrix.txt
#scp sskg8@mill.mst.edu:/mnt/stor/ceph/csc/sdas-lab/Shovan/adjacency_matrix100_p0.5.txt ~/Desktop/

