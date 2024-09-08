# Instruction to run the code
## Load modules
module load cuda-toolkit/12.5

## Build and run
nvcc  \<fileName.cu\> -o \<fileName\> && ./\<fileName\>
