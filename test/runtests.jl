using Test
using MPI
using JLD2

nprocs = 3
run(
    `$(mpiexec()) -n $nprocs $(Base.julia_cmd()) --project=$(Base.active_project()) test_mpi.jl`,
)
run(`$(mpiexec()) -n 1 $(Base.julia_cmd()) --project=$(Base.active_project()) test_mpi.jl`)

mktempdir() do dir
    outfile = dir * "/out.jld2"
    run(
        `$(mpiexec()) -n $nprocs $(Base.julia_cmd()) --project=$(Base.active_project()) test_mpi.jl $outfile`,
    )
    run(
        `$(mpiexec()) -n 1 $(Base.julia_cmd()) --project=$(Base.active_project()) test_mpi.jl $outfile`,
    )
end
