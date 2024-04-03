
@testset "Accumulators" begin
    size = 124

    mktempdir() do dir
        mem = MPIScheduler.MemoryAccumulator(size)
        jld = MPIScheduler.JLD2Accumulator(size, dir * "/test.jld2", 13)

        for acc in [mem, jld]
            @testset "$(typeof(acc))" begin
                is = unique(rand(1:size, 5))
                for i in is
                    acc[i] = 1 + i
                end
                @test collect(acc) == [i in is ? 1 + i : nothing for i = 1:size]
                MPIScheduler.flush!(acc)

                for i in Random.shuffle(1:size)
                    acc[i] = i
                end

                res = similar(acc)
                for i in Random.shuffle(1:size)
                    res[i] = acc[i]
                end

                @test res == collect(1:size)
            end
        end
    end
end
