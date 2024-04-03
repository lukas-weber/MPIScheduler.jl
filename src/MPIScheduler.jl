module MPIScheduler
using MPI
using Printf
using Dates
using JLD2

include("result_accumulator.jl")

const TAG_TASKID = 1345
const TAG_DONE = 1346

struct Silent end
struct Idle end

function run(
    funcs::AbstractVector;
    output_file::Union{Nothing,AbstractString} = nothing,
    comm = MPI.COMM_WORLD,
    log_frequency::Union{Silent,Integer} = max(1, length(funcs) ÷ 1000),
    accumulator_buffer_size = max(1, length(funcs) ÷ 50),
)
    accumulator =
        isnothing(output_file) ? MemoryAccumulator(length(funcs)) :
        JLD2Accumulator(length(funcs), output_file, accumulator_buffer_size)
    MPI.Init()
    if MPI.Comm_size(comm) == 1
        for (i, f) in enumerate(funcs)
            if !hasresult(accumulator, i)
                accumulator[i] = f()
            end
            log_progress(i, length(funcs), log_frequency)
        end
        return accumulator
    end

    MPI.Barrier(MPI.COMM_WORLD)
    if MPI.Comm_rank(comm) == 0
        return controller(accumulator, comm; log_frequency)
    else
        return worker(funcs, comm)
    end
end

function find_next_task(accumulator, inprogress, done)
    inprogress += 1
    while inprogress <= length(accumulator) && hasresult(accumulator, inprogress)
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

function controller(accumulator, comm; log_frequency)
    inprogress = 0
    done = 0

    active_workers = MPI.Comm_size(comm) - 1

    while active_workers > 0
        (taskid, result), status =
            MPI.recv(comm, MPI.Status; source = MPI.ANY_SOURCE, tag = TAG_DONE)

        if result !== Idle()
            accumulator[taskid] = result
            done += 1
            log_progress(done, length(accumulator), log_frequency)
        end

        inprogress, done = find_next_task(accumulator, inprogress, done)
        if inprogress > length(accumulator) # everything done
            active_workers -= 1
        end
        MPI.send(inprogress, comm; dest = status.source, tag = TAG_TASKID)
    end

    flush!(accumulator)
    MPI.Barrier(MPI.COMM_WORLD)
    return accumulator
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
