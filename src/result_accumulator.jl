abstract type AbstractResultAccumulator <: AbstractVector{Any} end
Base.size(a::AbstractResultAccumulator) = (a.full_length,)

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

function Base.setindex!(m::MemoryAccumulator, index::Integer, r)
    m.results[index] = r
    return nothing
end

Base.getindex(m::MemoryAccumulator, index::Integer) = m.results[index]


struct JLD2Accumulator <: AbstractResultAccumulator
    full_length::Int
    filename::String
end

function Base.setindex!(m::JLD2Accumulator, index::Integer, r)
    jldopen(m.filename, "a+") do f
        f[string(index)] = r
    end
end

Base.getindex(m::JLD2Accumulator, index::Integer) =
    jldopen(f -> f[string(index)], m.filename, "a+")
