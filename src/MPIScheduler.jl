module MPIScheduler
using MPI
using Printf
using Dates


export provide_pmap, Silent

include("pmap.jl")

const TAG_TASKID = 1345
const TAG_DONE = 1346

struct Silent end
struct Idle end

struct NoResult end

const CHUNK_SIZE = 1 << 31 - 1

@enum WorkerState begin
    WSSentWork
    WSSendingWork
    WSReceivingResult
    WSWaitingForResult
    WSDone
end

# from the perspective of the controller
mutable struct WorkerContext
    rank::Int
    state::WorkerState
    request::MPI.Request
    buffer::Vector{UInt8}
    total_bytes::Ref{Int64}
    transferred_bytes::Int64
    will_finish::Bool

    current_taskid::Int
end

function WorkerContext(rank, comm)
    total_bytes = Ref{Int64}(0)
    req = MPI.Irecv!(total_bytes, comm; source = rank, tag = TAG_DONE)
    return WorkerContext(rank, WSWaitingForResult, req, UInt8[], total_bytes, 0, false, 0)
end

isdone(w::WorkerContext) = w.state == WSDone

function handle!(w::WorkerContext, comm, next_task::Tuple{Int,Any})
    function irecv_next!(received, total, tag)
        i1 = received + 1
        i2 = min(i1 - 1 + CHUNK_SIZE, total)
        return MPI.Irecv!(view(w.buffer, i1:i2), comm; source = w.rank, tag)
    end

    # println("$(w.rank): $(w.state), $next_task")

    if w.state == WSSentWork
        if w.will_finish
            w.state = WSDone
            return nothing
        else
            w.state = WSWaitingForResult
            w.request = MPI.Irecv!(w.total_bytes, comm; source = w.rank, tag = TAG_DONE)
            return nothing
        end
    elseif w.state == WSWaitingForResult
        w.state = WSReceivingResult
        resize!(w.buffer, w.total_bytes[])
        w.transferred_bytes = 0
        w.request = irecv_next!(0, w.total_bytes[], TAG_DONE)
        return nothing
    elseif w.state == WSReceivingResult
        w.transferred_bytes += CHUNK_SIZE
        if w.transferred_bytes < w.total_bytes[] > 0
            w.request = irecv_next!(w.transferred_bytes, w.total_bytes[], TAG_DONE)
            return nothing
        else
            next_taskid, next_data = next_task
            old_taskid, w.current_taskid = w.current_taskid, next_taskid
            if isnothing(next_data)
                w.will_finish = true
            end
            w.state = WSSendingWork
            result = MPI.deserialize(w.buffer)
            s = MPI.serialize(next_data)
            if length(s) > length(w.buffer)
                w.buffer = s
            else
                view(w.buffer, eachindex(s)) .= s
            end
            w.total_bytes[] = length(s)
            w.transferred_bytes = 0
            w.request = MPI.Isend(w.total_bytes, comm; dest = w.rank, tag = TAG_TASKID)

            return old_taskid, result
        end
    elseif w.state == WSSendingWork
        w.request = MPI.Isend(
            view(
                w.buffer,
                1+w.transferred_bytes:min(
                    w.transferred_bytes + CHUNK_SIZE,
                    w.total_bytes[],
                ),
            ),
            comm;
            dest = w.rank,
            tag = TAG_TASKID,
        )
        w.transferred_bytes += CHUNK_SIZE
        if w.transferred_bytes >= w.total_bytes[]
            w.state = WSSentWork
        end
        return nothing
    end
end

function send_chunked(x, comm; dest, tag)
    chunks = Iterators.partition(MPI.serialize(x), CHUNK_SIZE)

    MPI.Send(sum(length, chunks), comm; dest, tag)
    for chunk in chunks
        MPI.Send(chunk, comm; dest, tag)
    end
end

function recv_chunked(comm; source, tag)
    total_length, status = MPI.Recv(Int64, comm, MPI.Status; source, tag)

    buf = zeros(UInt8, total_length) #Vector{UInt8}(undef, total_length)
    for chunk in Iterators.partition(buf, CHUNK_SIZE)
        MPI.Recv!(chunk, comm; status.source, tag)
    end
    return MPI.deserialize(buf), status
end


function run(
    funcs::Union{AbstractVector,Nothing};
    comm = MPI.COMM_WORLD,
    log_frequency::Union{Silent,Integer} = isnothing(funcs) ? Silent() :
                                           max(1, length(funcs) ÷ 1000),
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
        return controller(funcs, comm; log_frequency)
    else
        return worker(comm)
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

function wait_for_worker!(workers; tries = 3, min_sleep = 0.001, max_sleep = 0.02)
    i = MPI.Waitany([w.request for w in workers])
    return popat!(workers,i)
    sleeptime = min_sleep

    while true
        for _ = 1:tries
            for (i, w) in enumerate(workers)
                flag = MPI.Test(w.request)
                if flag
                    return popat!(workers, i)
                end
            end
        end
        tries = 1
        sleep(sleeptime)
        if sleeptime < max_sleep
            sleeptime = min(max_sleep, sleeptime * 1.5)
        end
    end
end

function controller(funcs, comm; log_frequency)
    num_tasks = length(funcs)
    results = Any[NoResult() for _ = 1:num_tasks]
    inprogress = 1
    done = 0

    workers = WorkerContext.(1:MPI.Comm_size(comm)-1, Ref(comm))

    while !isempty(workers)
        next_worker = wait_for_worker!(workers)

        r = handle!(next_worker, comm, (inprogress, get(funcs, inprogress, nothing)))
        if r !== nothing
            (taskid, result) = r

            if result !== Idle()
                results[taskid] = result
                done += 1
                log_progress(done, num_tasks, log_frequency)
            end

            inprogress, done = find_next_task(results, inprogress, done)
        end

        if !isdone(next_worker)
            push!(workers, next_worker)
        end
    end

    MPI.Barrier(MPI.COMM_WORLD)
    return results
end

function worker(comm)
    send_chunked(Idle(), comm; dest = 0, tag = TAG_DONE)

    while true
        waittime = @elapsed begin
            func, _ = recv_chunked(comm; source = 0, tag = TAG_TASKID)
        end
        if waittime > 1
            @warn "$(MPI.Comm_rank(comm)) waited a long time for a new task: $waittime s"
        end
        if isnothing(func)
            break
        end

        result = @invokelatest func()
        send_chunked(result, comm; dest = 0, tag = TAG_DONE)
    end

    MPI.Barrier(MPI.COMM_WORLD)
    return nothing
end

end # module MPIScheduler
