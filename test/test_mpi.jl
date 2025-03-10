using Test
using MPI
using MPIScheduler

funcs(L) = [() -> i for i = 1:L]
Ls = 1:10

MPI.Init()
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @testset "nranks = $(MPI.Comm_size(MPI.COMM_WORLD))" begin
        @testset for L in Ls
            results = MPIScheduler.run(funcs(L); log_frequency = MPIScheduler.Silent())
            @test results == [f() for f in funcs(L)]
            provide_pmap(; log_frequency = Silent()) do pmap
                @test pmap(x -> x^2, 1:L) == (1:L) .^ 2
            end
        end
    end
else
    for L in Ls
        MPIScheduler.run(funcs(L))
        provide_pmap(; log_frequency = Silent()) do pmap
            @test pmap(x -> x^2, 1:L) == (1:L) .^ 2
        end
    end
end
