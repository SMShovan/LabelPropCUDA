import cudf
import cugraph
from cugraph.components import connected_components

import sys

# Ensure correct number of arguments
if len(sys.argv) != 3:
    print("Usage: python process_connected_components.py <input_file> <output_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Load the input graph from the file
try:
    edges_df = cudf.read_csv(input_file, delimiter=" ", header=None, names=["src", "dst", "weight"])
    print("Input Edges DataFrame:")
    print(edges_df)
except Exception as e:
    print(f"Error reading input file {input_file}: {e}")
    sys.exit(1)

# Create a Graph in cuGraph
G = cugraph.Graph()
G.from_cudf_edgelist(edges_df, source='src', destination='dst', edge_attr='weight', renumber=True)

# Compute the connected components
cc_df = connected_components(G)

# Save the results to the output file
try:
    cc_df.to_csv(output_file, index=False)
    print(f"Connected components saved to {output_file}")
except Exception as e:
    print(f"Error writing output file {output_file}: {e}")
    sys.exit(1)
