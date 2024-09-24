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

    arg_prefixes = ntuple(i -> collect_arg_addresses((i,), args[i]), length(args))

    return vcat([(; prefix, stage, func = f, args = arg_prefixes)], arg_entries...)
end

collect_entries(prefix::Tuple, node::Function) = collect_entries(prefix, (node,))

function collect_arg_addresses(prefix::Tuple, node::AbstractVector)
    return [collect_arg_addresses((prefix..., i), a) for (i, a) in enumerate(node)]
end

collect_arg_addresses(prefix::Tuple, node::Union{Function,Tuple}) = prefix

construct_arg(results::AbstractDict, prefix::Tuple, arg_addresses::AbstractVector) =
    [construct_arg(results, prefix, arg) for arg in arg_addresses]
construct_arg(results::AbstractDict, prefix::Tuple, arg_address::Tuple) =
    results[(prefix..., arg_address...)]

function delete_arg!(results::AbstractDict, prefix::Tuple, arg_addresses::AbstractVector)
    for arg in arg_addresses
        delete_arg!(results, prefix, arg)
    end
end

delete_arg!(results::AbstractDict, prefix::Tuple, arg_address::Tuple) =
    delete!(results, (prefix..., arg_address...))

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
                () -> e.func(construct_arg.(Ref(results), Ref(e.prefix), e.args)...) for
                e in stage_entries
            ];
            kwargs...,
        )

        for (entry, result) in zip(stage_entries, stage_results)
            for arg in entry.args
                delete_arg!(results, entry.prefix, arg)
            end
            results[entry.prefix] = result
        end
    end

    if length(results) == 1
        return results[()]
    end
    return construct_arg(results, (), collect_arg_addresses((), tree))
end
