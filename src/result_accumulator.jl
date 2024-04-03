abstract type AbstractResultAccumulator <: AbstractVector{Any} end
Base.size(a::AbstractResultAccumulator) = (a.full_length,)
Base.IndexStyle(::Type{<:AbstractResultAccumulator}) = IndexLinear()

hasresult(a::AbstractResultAccumulator, i::Integer) = !isnothing(a[i])

struct MemoryAccumulator <: AbstractResultAccumulator
    full_length::Int
    results::Dict{Int,Any}
end

MemoryAccumulator(full_length::Integer) = MemoryAccumulator(full_length, Dict{Int,Any}())

flush!(::MemoryAccumulator) = nothing

Base.setindex!(m::MemoryAccumulator, r, index::Integer) = (m.results[index] = r)
Base.getindex(m::MemoryAccumulator, index::Integer) = get(m.results, index, nothing)


struct JLD2Accumulator <: AbstractResultAccumulator
    full_length::Int
    filename::String

    buffer::Dict{Int,Tuple{Bool,Any}}
    buffer_size::Int
end

JLD2Accumulator(full_length::Integer, filename::AbstractString, buffer_size::Integer) =
    JLD2Accumulator(full_length, filename, Dict{Int,Any}(), buffer_size)

idx2string(i, full_length) = lpad(i, ceil(Int, log10(full_length + 0.5)), '0')

flush!(m::JLD2Accumulator) = jldopen(f -> flush!(m, f), m.filename, "a+")

function flush!(m::JLD2Accumulator, file_handle::JLD2.JLDFile)
    for (i, (mutated, r)) in m.buffer
        key = idx2string(i, m.full_length)
        if mutated && r !== nothing
            if haskey(file_handle, key)
                delete!(file_handle, key)
            end
            file_handle[key] = r
        end
    end
    empty!(m.buffer)
end

function Base.setindex!(m::JLD2Accumulator, r, index::Integer)
    m.buffer[index] = (true, r)

    if length(m.buffer) > m.buffer_size
        flush!(m)
    end

    return r
end

function read_buffer!(m::JLD2Accumulator, index)
    page = fld1(index, m.buffer_size) - 1
    jldopen(m.filename, "a+") do f
        flush!(m)
        for i = 1:m.buffer_size
            idx = m.buffer_size * page + i
            if idx > m.full_length
                break
            end
            m.buffer[idx] = (false, get(f, idx2string(idx, m.full_length), nothing))
        end
    end
    @assert haskey(m.buffer, index)
end

function Base.getindex(m::JLD2Accumulator, index::Integer)
    if !haskey(m.buffer, index)
        read_buffer!(m, index)
    end

    return m.buffer[index][2]
end
