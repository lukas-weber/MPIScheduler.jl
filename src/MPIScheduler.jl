module MPIScheduler
using MPI
using ProgressMeter

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

function run(funcs::AbstractVector; comm = MPI.COMM_WORLD)
    MPI.Init()

    res = nothing
    if MPI.Comm_rank(comm) == 0
        t_work = @async res = worker($funcs, $comm)
        t_ctrl = @async controller($funcs, $comm)
        sync_or_error([t_work, t_ctrl])

        return getindex.(Ref(reduce(merge,MPI.gather(res, comm))), eachindex(funcs))
    else
        res = worker(funcs, comm) 
        MPI.gather(res, comm)
        return nothing
    end
end

function controller(funcs, comm)
    inprogress = 0
    done = 0
    for i = 1:min(length(funcs), MPI.Comm_size(comm))
        send(i, comm; dest=i-1, tag=TAG_TASKID)
        inprogress += 1
    end

    p = Progress(length(funcs))

    while done < length(funcs)
        _, status = recv(Int, comm; source = MPI.ANY_SOURCE, tag=TAG_DONE)

        done += 1
        next!(p)
        send(inprogress+1, comm; dest = status.source, tag = TAG_TASKID)
        inprogress += 1
    end

    return nothing
end      

function worker(funcs, comm)
    results = Dict{Int, Any}()
    while true
        taskid,_ = recv(Int, comm; source = 0, tag=TAG_TASKID)
        if taskid < 1 || taskid > length(funcs)
            break
        end        

        results[taskid] = funcs[taskid]()
        send(taskid, comm; dest = 0, tag=TAG_DONE)
    end

    return results
end

end # module MPIScheduler
