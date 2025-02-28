using Test
using MPI
using MPIScheduler

nprocs = 3
run(
    `$(mpiexec()) -n $nprocs $(Base.julia_cmd()) --project=$(Base.active_project()) test_mpi.jl`,
)
run(`$(mpiexec()) -n 1 $(Base.julia_cmd()) --project=$(Base.active_project()) test_mpi.jl`)
