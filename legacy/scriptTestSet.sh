import os

# Define constant parameters
probability = 0.7
sRand = 1
eRand = 10
output_dir = "/mnt/stor/ceph/csc/sdas-lab/Shovan/"

# Iterate over node values from 1000 to 1000000 in multiples of 10
nodes_values = [10**i for i in range(3, 7)]
for nodes in nodes_values:
    output_file = os.path.join(output_dir, f"adjacency_matrix{nodes}.txt")
    command = (
        f"python generate_test_set.py --probability {probability} "
        f"--nodes {nodes} --sRand {sRand} --eRand {eRand} --output {output_file}"
    )
    print(f"Executing: {command}")
    os.system(command)
