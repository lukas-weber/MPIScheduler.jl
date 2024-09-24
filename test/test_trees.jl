
function get_prefix_stages(tree)
    return Set((e.prefix, e.stage) for e in MPIScheduler.collect_entries((), tree))
end

@testset "Trees" begin

    f = Returns(1)
    tree1 = [f, f, (f,)]
    @test get_prefix_stages(tree1) == Set([((1,), 1), ((2,), 1), ((3,), 1)])
    @test MPIScheduler.run_tree(tree1) == [1, 1, 1]


    tree2 = (+, f, f, (*, f, f))

    @test get_prefix_stages(tree2) ==
          Set([((), 3), ((1,), 1), ((2,), 1), ((3,), 2), ((3, 1), 1), ((3, 2), 1)])
    @test MPIScheduler.run_tree(tree2) == 3
end
