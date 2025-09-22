module MPIScheduler
using MPI
using Printf
using Dates
using Serialization


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

struct WorkerStats
    recv_time::Float64
    send_time::Float64
    serialization_time::Float64
    total_time::Float64
    num_tasks::Int
end

# from the perspective of the controller
mutable struct WorkerContext
    rank::Int
    state::WorkerState
    request::MPI.Request
    buffer::IOBuffer
    total_bytes::Ref{Int64}
    transferred_bytes::Int64
    will_finish::Bool

    current_taskid::Int
end

function WorkerContext(rank, comm)
    total_bytes = Ref{Int64}(0)
    req = MPI.Irecv!(total_bytes, comm; source = rank, tag = TAG_DONE)
    return WorkerContext(
        rank,
        WSWaitingForResult,
        req,
        IOBuffer(),
        total_bytes,
        0,
        false,
        0,
    )
end

isdone(w::WorkerContext) = w.state == WSDone

function handle!(w::WorkerContext, comm, next_task::Tuple{Int,Any})
    function irecv_next!(received, total, tag)
        i1 = received + 1
        i2 = min(i1 - 1 + CHUNK_SIZE, total)
        return MPI.Irecv!(view(w.buffer.data, i1:i2), comm; source = w.rank, tag)
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
        truncate(w.buffer, w.total_bytes[])
        seekstart(w.buffer)
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
            result = deserialize(w.buffer)
            seekstart(w.buffer)
            serialize(w.buffer, next_data)
            w.total_bytes[] = position(w.buffer)
            w.transferred_bytes = 0
            w.request = MPI.Isend(w.total_bytes, comm; dest = w.rank, tag = TAG_TASKID)

            return old_taskid, result
        end
    elseif w.state == WSSendingWork
        w.request = MPI.Isend(
            view(
                w.buffer.data,
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

function send_chunked(buf, comm; dest, tag)
    chunks = Iterators.partition(buf, CHUNK_SIZE)

    MPI.Send(sum(length, chunks), comm; dest, tag)
    for chunk in chunks
        MPI.Send(chunk, comm; dest, tag)
    end
    return nothing
end

function recv_chunked(comm; source, tag)
    total_length, status = MPI.Recv(Int64, comm, MPI.Status; source, tag)

    buf = zeros(UInt8, total_length) #Vector{UInt8}(undef, total_length)
    for chunk in Iterators.partition(buf, CHUNK_SIZE)
        MPI.Recv!(chunk, comm; status.source, tag)
    end
    return buf, status
end


function run(
    funcs::Union{AbstractVector,Nothing};
    comm = MPI.COMM_WORLD,
    log_frequency::Union{Silent,Integer} = isnothing(funcs) ? Silent() :
                                           max(1, length(funcs) ÷ 1000),
    log_advanced_stats = true,
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

    MPI.Barrier(comm)
    if MPI.Comm_rank(comm) == 0
        if length(funcs) == 1
            controller([], comm; log_frequency = Silent(), log_advanced_stats = false)
            return [funcs[1]()]
        end
        return controller(funcs, comm; log_frequency, log_advanced_stats)
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
        @info "$(round(now(), Dates.Second)): $(@sprintf("%5d/%d", done, num_tasks))"
    end

    return nothing
end

function print_advanced_stats(ctl_wait_time, ctl_handle_time, ctl_total_time, worker_stats)
    @info @sprintf(
        "controller: %.2f%% waiting %.2f%% handling of %.2f s total time",
        100 * ctl_wait_time / ctl_total_time,
        100 * ctl_handle_time / ctl_total_time,
        ctl_total_time
    )

    worker_times =
        sum(worker_stats) do stats
            [stats.recv_time, stats.send_time, stats.serialization_time] /
            stats.total_time
        end / length(worker_stats)

    total_efficiency =
        sum(worker_stats) do stats
            stats.total_time - stats.recv_time - stats.send_time -
            stats.serialization_time
        end / (length(worker_stats) * ctl_total_time)

    @info @sprintf(
        "workers: %.2f%% sending %.2f%% receiving %.2f%% serializing",
        100 .* worker_times...
    )
    @info @sprintf("parallel efficiency: %.2f%%", 100 .* total_efficiency)

    return nothing
end

function controller(funcs, comm; log_frequency, log_advanced_stats)
    num_tasks = length(funcs)
    results = Any[NoResult() for _ = 1:num_tasks]
    inprogress = 1
    done = 0

    workers = WorkerContext.(1:MPI.Comm_size(comm)-1, Ref(comm))

    wait_time = 0.0
    handle_time = 0.0

    total_time = @elapsed while !isempty(workers)
        wait_time += @elapsed i = MPI.Waitany([w.request for w in workers])
        next_worker = workers[i]

        handle_time += @elapsed begin
            GC.enable(false)
            r = handle!(next_worker, comm, (inprogress, get(funcs, inprogress, nothing)))
            GC.enable(true)
        end
        if r !== nothing
            (taskid, result) = r

            if result !== Idle()
                results[taskid] = result
                done += 1
                log_progress(done, num_tasks, log_frequency)
            end

            inprogress, done = find_next_task(results, inprogress, done)
        end

        if isdone(next_worker)
            popat!(workers, i)
        end
    end


    worker_stats = MPI.Gather([WorkerStats(0, 0, 0, 0, 0)], comm)[2:end]
    if log_advanced_stats
        print_advanced_stats(wait_time, handle_time, total_time, worker_stats)
    end

    return results
end

function worker(comm)
    send_chunked(MPI.serialize(Idle()), comm; dest = 0, tag = TAG_DONE)

    num_tasks = 0

    recv_time = 0.0
    send_time = 0.0
    serialization_time = 0.0
    total_time = @elapsed while true
        recv_time += @elapsed begin
            buf, _ = recv_chunked(comm; source = 0, tag = TAG_TASKID)
        end
        serialization_time += @elapsed func = MPI.deserialize(buf)
        if isnothing(func)
            break
        end
        num_tasks += 1

        result = @invokelatest func()
        serialization_time += @elapsed buf = MPI.serialize(result)
        send_time += @elapsed send_chunked(buf, comm; dest = 0, tag = TAG_DONE)
    end
    efficiency = 1 - (send_time + recv_time) / total_time

    serialization = serialization_time / total_time


    MPI.Gather(
        [WorkerStats(recv_time, send_time, serialization_time, total_time, num_tasks)],
        comm,
    )

    return nothing
end

end # module MPIScheduler
