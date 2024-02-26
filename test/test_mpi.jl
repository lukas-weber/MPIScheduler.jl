using Test
using MPI
using MPIScheduler

funcs(L) = [() -> i for i = 1:L]
Ls = 1:10

MPI.Init()
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @testset "run" begin
        @testset for L in Ls
            results = MPIScheduler.run(funcs(L))
            @test results == [f() for f in funcs(L)]
        end
    end
else
    for L in Ls
        MPIScheduler.run(funcs(L))
    end
end
