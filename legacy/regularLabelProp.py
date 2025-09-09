import numpy as np
import networkx as nx

def custom_label_propagation_average(G, tol=1e-6):
    """
    Update each node's label to the average of its neighbors' labels until convergence.
    Returns the number of iterations until convergence.
    """
    # Initialize each node with its own label (as a float)
    labels = {node: float(node) for node in G.nodes()}
    iteration = 0
    changed = True

    while changed:
        iteration += 1
        changed = False
        
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue  # Skip isolated nodes
            
            # Compute the average label of the neighbors
            neighbor_labels = [labels[neighbor] for neighbor in neighbors]
            avg_label = sum(neighbor_labels) / len(neighbor_labels)
            
            # If the change is significant, update the label
            if abs(avg_label - labels[node]) > tol:
                labels[node] = avg_label
                changed = True

    return iteration

def get_iterations(filename="adjacency_matrix1000_p0.1.txt"):
    """
    Loads an adjacency matrix from a file, creates a graph,
    runs the custom label propagation algorithm, and returns
    the number of iterations until convergence.
    """
    # Adjust delimiter if needed (e.g., delimiter="," for CSV files)
    adj_matrix = np.loadtxt(filename, delimiter=None)
    G = nx.from_numpy_array(adj_matrix)
    iterations = custom_label_propagation_average(G)
    return iterations

def main():
    filename = "/mnt/stor/ceph/csc/sdas-lab/Shovan/adjacency_matrix10000_p0.1.txt"  # Change this to your file
    iterations = get_iterations(filename)
    print(f"Number of iterations until convergence: {iterations}")

if __name__ == "__main__":
    main()
