
function provide_pmap(func; comm = MPI.COMM_WORLD, kwargs...)
    MPI.Init()

    if MPI.Comm_size(comm) == 1
        return func(map)
    end

    function pmap(f, args)
        MPI.Bcast(true, 0, comm)

        funcs = [() -> @invokelatest(f(a)) for a in args]
        return run(funcs; comm, kwargs...)
    end

    if MPI.Comm_rank(comm) == 0
        rv = func(pmap)

        MPI.Bcast(false, 0, comm)
        return rv
    end

    # workers listen for pmap calls
    while true
        another_one = MPI.Bcast(true, 0, comm)
        if !another_one
            break
        end
        run(nothing; comm, kwargs...)
    end
    return nothing
end
