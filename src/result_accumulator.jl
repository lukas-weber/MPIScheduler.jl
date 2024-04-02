abstract type AbstractResultAccumulator <: AbstractVector{Any} end
Base.size(a::AbstractResultAccumulator) = (a.full_length,)
Base.IndexStyle(::Type{<:AbstractResultAccumulator}) = IndexLinear()

function hasresult(a::AbstractResultAccumulator, i::Integer)
    try
        a[i]
    catch
        (KeyError)
        return false
    end
    return true
end

struct MemoryAccumulator <: AbstractResultAccumulator
    full_length::Int
    results::Dict{Int,Any}
end

MemoryAccumulator(full_length::Integer) = MemoryAccumulator(full_length, Dict{Int,Any}())

flush!(::MemoryAccumulator) = nothing

function Base.setindex!(m::MemoryAccumulator, r, index::Integer)
    m.results[index] = r
    return nothing
end

Base.getindex(m::MemoryAccumulator, index::Integer) = m.results[index]


struct JLD2Accumulator <: AbstractResultAccumulator
    full_length::Int
    filename::String

    buffer::Dict{Int,Any}
    buffer_size::Int
end

JLD2Accumulator(full_length::Integer, filename::AbstractString, buffer_size::Integer) =
    JLD2Accumulator(full_length, filename, Dict{Int,Any}(), buffer_size)

idx2string(i, full_length) = lpad(i, ceil(Int, log10(full_length + 0.5)), '0')

function flush!(m::JLD2Accumulator)
    jldopen(m.filename, "a+") do f
        for (i, r) in m.buffer
            f[idx2string(i, m.full_length)] = r
        end
    end
    empty!(m.buffer)
end

function Base.setindex!(m::JLD2Accumulator, r, index::Integer)
    m.buffer[index] = r

    if length(m.buffer) >= m.buffer_size
        flush!(m)
    end
end

Base.getindex(m::JLD2Accumulator, index::Integer) =
    jldopen(f -> f[idx2string(index, m.full_length)], m.filename, "a+")
