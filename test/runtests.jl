using Test
using MPI
using MPIScheduler

@testset "MPI" begin
    funcs = [() -> i for i = 1:10]

    nprocs = 3
    run(
        `$(mpiexec()) -n $nprocs $(Base.julia_cmd()) --project=$(Base.active_project()) test_mpi.jl`,
    )
end
