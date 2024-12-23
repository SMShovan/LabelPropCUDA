import sys
import cupy as cp

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 rowSum.py <input_file> <output_file>")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Load the sparse graph from the input file
    with open(input_file, 'r') as f:
        graph = [list(map(int, line.strip().split())) for line in f]

    # Convert to a CuPy array
    graph_gpu = cp.array(graph, dtype=cp.int32)

    # Compute the row sums
    row_sums = cp.sum(graph_gpu, axis=1)

    # Save the row sums to the output file
    with open(output_file, 'w') as f:
        for row_sum in row_sums.get():  # Transfer data back to CPU
            f.write(f"{row_sum}\n")

if __name__ == "__main__":
    main()
