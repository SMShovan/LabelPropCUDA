import os

# Define constant parameters
sRand = 1
eRand = 10
output_dir = "/mnt/stor/ceph/csc/sdas-lab/Shovan/"

# Iterate over node values from 1000 to 1000000 in multiples of 10
nodes_values = [10**i for i in range(3, 8)]
probability_values = [round(0.1 + 0.2 * i, 1) for i in range(5)]  # 0.1 to 0.9 with step size 0.2

for nodes in nodes_values:
    for probability in probability_values:
        output_file = os.path.join(output_dir, f"adjacency_matrix{nodes}_p{probability}.txt")
        command = (
            f"python generate_test_set.py --probability {probability} "
            f"--nodes {nodes} --sRand {sRand} --eRand {eRand} --output {output_file}"
        )
        print(f"Executing: {command}")
        os.system(command)
