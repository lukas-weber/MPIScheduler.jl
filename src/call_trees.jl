function collect_entries(prefix::Tuple, node::AbstractVector)
    return mapreduce(vcat, enumerate(node)) do ientry
        (i, entry) = ientry
        return collect_entries((prefix..., i), entry)
    end
end

function collect_entries(prefix::Tuple, node::Tuple)
    (f, args...) = node

    arg_entries = ntuple(i -> collect_entries((prefix..., i), args[i]), length(args))

    stage = maximum(entry -> entry.stage, Iterators.flatten(arg_entries), init = 0) + 1

    return vcat([(; prefix, stage, func = f, num_args = length(args))], arg_entries...)
end

collect_entries(prefix::Tuple, node::Function) = collect_entries(prefix, (node,))

"""
    run_tree(tree::Tuple)

Runs a call tree, in which each node is either
    - a function
    - a vector of nodes
    - a tuple of the form `(f, args...)` where `f` is a function and each element of `args` is a node.
"""
function run_tree(tree; kwargs...)
    entries = collect_entries((), tree)

    sort!(entries, by = e -> e.stage)
    num_stages = entries[end].stage

    results = Dict()
    entry_queue = entries
    for stage = 1:num_stages
        stage_entries = collect(Iterators.takewhile(e -> e.stage == stage, entry_queue))
        entry_queue = Iterators.drop(entry_queue, length(stage_entries))

        stage_results = MPIScheduler.run(
            [
                () -> e.func((results[(e.prefix..., i)] for i = 1:e.num_args)...) for
                e in stage_entries
            ];
            kwargs...,
        )

        for (entry, result) in zip(stage_entries, stage_results)
            for n = 1:entry.num_args
                delete!(results, (entry.prefix..., n))
            end
            results[entry.prefix] = result
        end
    end

    if length(results) == 1
        return results[()]
    end
    return [results[(i,)] for i = 1:length(results)]
end
