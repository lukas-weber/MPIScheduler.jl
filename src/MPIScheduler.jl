module MPIScheduler
using MPI
using Printf
using Dates

const TAG_TASKID = 1345
const TAG_DONE = 1346

function send(data, comm; dest, tag)
    req = MPI.Isend(data, comm; dest, tag)
    while !MPI.Test(req)
        yield()
    end
    return nothing
end

function recv(::Type{T}, comm; source, tag) where {T}
    data = Ref{T}()
    req = MPI.Irecv!(data, comm; source = source, tag = tag)
    status = MPI.Status(0, 0, 0, 0, 0)

    while ((flag, status) = MPI.Test(req, MPI.Status); !flag)
        yield()
    end
    return data[], status
end

struct TaskInterruptedException <: Exception end

# Base.@sync only propagates errors once all tasks are done. We want
# to fail everything as soon as one task is broken. Possibly this is
# not completely bullet-proof, but good enough for now.
function sync_or_error(tasks::AbstractArray{Task})
    c = Channel(Inf)
    for t in tasks
        @async begin
            Base._wait(t)
            put!(c, t)
        end
    end
    for _ in eachindex(tasks)
        t = take!(c)
        if istaskfailed(t)
            for tother in tasks
                if tother != t
                    schedule(tother, TaskInterruptedException(); error = true)
                end
            end
            throw(TaskFailedException(t))
        end
    end
    close(c)
end

function run(
    funcs::AbstractVector;
    comm = MPI.COMM_WORLD,
    log_frequency = max(1, length(funcs) ÷ 1000),
)
    MPI.Init()

    if MPI.Comm_rank(comm) == 0
        # t_work = @async res = worker($funcs, $comm)
        # t_ctrl = @async controller($funcs, $comm)
        # sync_or_error([t_work, t_ctrl])
        controller(funcs, comm; log_frequency)
        res = Dict()
        return getindex.(Ref(reduce(merge, MPI.gather(res, comm))), eachindex(funcs))
    else
        res = worker(funcs, comm)
        MPI.gather(res, comm)
        return nothing
    end
end

function controller(funcs, comm; log_frequency)
    inprogress = 0
    done = 0
    for i = 1:MPI.Comm_size(comm)-1
        send(i, comm; dest = i, tag = TAG_TASKID)
        inprogress += 1
    end

    while done < length(funcs)
        _, status = recv(Int, comm; source = MPI.ANY_SOURCE, tag = TAG_DONE)

        done += 1

        if done % log_frequency == 0 || done == length(funcs)
            println(
                "$(round(now(), Dates.Second)): $(@sprintf("%5d/%d", done, length(funcs)))",
            )
        end
        send(inprogress + 1, comm; dest = status.source, tag = TAG_TASKID)
        inprogress += 1
    end

    return nothing
end

function worker(funcs, comm)
    results = Dict{Int,Any}()
    while true
        waittime = @elapsed begin
            taskid, _ = recv(Int, comm; source = 0, tag = TAG_TASKID)
        end
        if waittime > 1
            @warn "waited a long time for a new task: $waittime s"
        end
        if taskid < 1 || taskid > length(funcs)
            break
        end

        results[taskid] = funcs[taskid]()
        send(taskid, comm; dest = 0, tag = TAG_DONE)
    end

    return results
end

end # module MPIScheduler
