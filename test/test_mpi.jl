using Test
using JLD2
using MPI
using MPIScheduler

funcs(L) = [() -> i for i = 1:L]
Ls = 1:10

output_file = length(ARGS) < 1 ? nothing : ARGS[1]

MPI.Init()
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @testset "nranks = $(MPI.Comm_size(MPI.COMM_WORLD))" begin
        @testset for L in Ls
            if !isnothing(output_file) && L > Ls[1]
                jldopen(output_file, "a+") do f
                    for i in unique(rand(keys(f), 4))
                        delete!(f, i)
                    end
                end
            end
            results = MPIScheduler.run(
                funcs(L);
                output_file,
                log_frequency = MPIScheduler.Silent(),
            )
            @test results == [f() for f in funcs(L)]
        end
    end
else
    for L in Ls
        MPIScheduler.run(funcs(L))
    end
end
