# Parallel-Computing-Project-2

compile and run sequential implementation:\
compile:\
g++ src/sequential_implementation.cpp -o sequential_implementation.exe\
run:\
./sequential_implementation.exe

compile and run mpi parallelization:\
compile:\
mpic++ src/mpi_parallelization.cpp -o mpi_parallelization.exe\
run:\
mpirun -np 4 ./mpi_parallelization.exe

to run python data analysis script:
python3 src/data_analysis.py
