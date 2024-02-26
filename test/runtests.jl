using Test
using MPI

nprocs = 3
run(
    `$(mpiexec()) -n $nprocs $(Base.julia_cmd()) --project=$(Base.active_project()) test_mpi.jl`,
)
run(
    `$(mpiexec()) -n 1 $(Base.julia_cmd()) --project=$(Base.active_project()) test_mpi.jl`,
)
