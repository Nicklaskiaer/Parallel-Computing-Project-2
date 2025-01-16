# Parallel-Computing-Project-2

compile and run sequential implementation:
compile:
g++ src/sequential_implementation.cpp -o sequential_implementation.exe
run:
./sequential_implementation.exe

compile and run mpi parallelization 1:
compile:
mpic++ src/mpi_parallelization.cpp -o mpi_parallelization.exe
run:
mpirun -np 4 ./mpi_parallelization.exe

compile and run mpi parallelization 2:
compile:
mpic++ src/mpi_parallelization_2.cpp -o mpi_parallelization_2.exe
run:
mpirun -np 4 ./mpi_parallelization_2.exe
