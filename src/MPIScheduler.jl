module MPIScheduler
using MPI
using Printf
using Dates

export provide_pmap, Silent

const TAG_TASKID = 1345
const TAG_DONE = 1346

struct Silent end
struct Idle end

struct NoResult end

function provide_pmap(func; comm = MPI.COMM_WORLD, kwargs...)
    MPI.Init()

    if MPI.Comm_size(comm) == 1
        return func(map)
    end

    function pmap(f, args)
        MPI.Bcast(true, 0, comm)

        MPI.bcast((f, args), comm)
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
        (f, args) = MPI.bcast(nothing, comm)
        funcs = [() -> @invokelatest(f(a)) for a in args]
        run(funcs; comm, kwargs...)
    end
end

function run(
    funcs::AbstractVector;
    comm = MPI.COMM_WORLD,
    log_frequency::Union{Silent,Integer} = max(1, length(funcs) ÷ 1000),
)
    MPI.Init()
    if MPI.Comm_size(comm) == 1
        results = Any[NoResult() for _ in eachindex(funcs)]
        for (i, f) in enumerate(funcs)
            results[i] = f()
            log_progress(i, length(funcs), log_frequency)
        end
        return results
    end

    MPI.Barrier(MPI.COMM_WORLD)
    if MPI.Comm_rank(comm) == 0
        return controller(length(funcs), comm; log_frequency)
    else
        return worker(funcs, comm)
    end
end

hasresult(results::AbstractVector, idx) = results[idx] !== NoResult()

function find_next_task(results, inprogress, done)
    inprogress += 1
    while inprogress <= length(results) && hasresult(results, inprogress)
        inprogress += 1
        done += 1
    end

    return inprogress, done
end

function log_progress(done, num_tasks, log_frequency)
    if log_frequency !== Silent() && (done % log_frequency == 0 || done == num_tasks)
        println("$(round(now(), Dates.Second)): $(@sprintf("%5d/%d", done, num_tasks))")
    end

    return nothing
end

function controller(num_tasks, comm; log_frequency)
    results = Any[NoResult() for _ = 1:num_tasks]
    inprogress = 0
    done = 0

    active_workers = MPI.Comm_size(comm) - 1

    while active_workers > 0
        (taskid, result), status =
            MPI.recv(comm, MPI.Status; source = MPI.ANY_SOURCE, tag = TAG_DONE)

        if result !== Idle()
            results[taskid] = result
            done += 1
            log_progress(done, num_tasks, log_frequency)
        end

        inprogress, done = find_next_task(results, inprogress, done)
        if inprogress > num_tasks # everything done
            active_workers -= 1
        end
        MPI.send(inprogress, comm; dest = status.source, tag = TAG_TASKID)
    end

    MPI.Barrier(MPI.COMM_WORLD)
    return results
end

function worker(funcs, comm)
    MPI.send((0, Idle()), comm; dest = 0, tag = TAG_DONE)

    while true
        waittime = @elapsed begin
            taskid = MPI.recv(comm; source = 0, tag = TAG_TASKID)
        end
        if waittime > 1
            @warn "waited a long time for a new task: $waittime s"
        end
        if taskid < 1 || taskid > length(funcs)
            break
        end

        result = funcs[taskid]()
        MPI.send((taskid, result), comm; dest = 0, tag = TAG_DONE)
    end

    MPI.Barrier(MPI.COMM_WORLD)
    return nothing
end

end # module MPIScheduler
