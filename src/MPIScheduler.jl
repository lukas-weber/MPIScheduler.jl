module MPIScheduler
using MPI
using Printf
using Dates

const TAG_TASKID = 1345
const TAG_DONE = 1346

struct Silent end

function send(data, comm; dest, tag)
    MPI.Send(data, comm; dest, tag)
    return nothing
end

function recv(::Type{T}, comm; source, tag) where {T}
    data, status = MPI.Recv(T, comm, MPI.Status; source, tag)

    return data, status
end

function run(
    funcs::AbstractVector;
    comm = MPI.COMM_WORLD,
    log_frequency::Union{Silent, Integer} = max(1, length(funcs) ÷ 1000),
)
    MPI.Init()
    if MPI.Comm_size(comm) == 1
        return [f() for f in funcs]
    end

    if MPI.Comm_rank(comm) == 0
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

        if log_frequency !== Silent() && (done % log_frequency == 0 || done == length(funcs))
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
