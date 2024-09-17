import matplotlib.pyplot as plt  # Importing the plotting library
import networkx as nx  # Importing networkx for graph operations

# Read the edges from graph.txt and treat the graph as undirected
edges_from_file = []
with open("tree.txt", "r") as file:
    for line in file:
        u, v, w1, w2 = map(int, line.split())
        edges_from_file.append((u, v, [w1, w2]))
        # Since the graph is undirected, add both (u, v) and (v, u)
        edges_from_file.append((v, u, [w1, w2]))

# Create an undirected graph
G_file = nx.Graph()  # Use Graph() for an undirected graph

# Add edges with their weights
for u, v, weight in edges_from_file:
    avg_weight = sum(weight) / 2  # Average of the two weights for layout purposes
    G_file.add_edge(u, v, weight=avg_weight)

# Draw the graph with labels
pos = nx.spring_layout(G_file)  # Positioning of nodes
nx.draw(G_file, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_color='black', font_weight='bold')

# Add edge labels with the weights (weight1, weight2)
edge_labels = {(u, v): f"{w[0]}, {w[1]}" for u, v, w in edges_from_file}
nx.draw_networkx_edge_labels(G_file, pos, edge_labels=edge_labels)

# Save the undirected graph as a PDF file
plt.title("Undirected Graph Visualization with NetworkX")
plt.savefig("undirected_graph_visualization.pdf", format="pdf")

# Clear the plot (optional, in case you want to create more plots later)
plt.clf()
