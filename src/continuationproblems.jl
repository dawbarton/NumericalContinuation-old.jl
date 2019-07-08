#-------------------------------------------------------------------------------
abstract type AbstractContinuationProblem{T} end

mutable struct ContinuationProblem{T, A, Z} <: AbstractContinuationProblem{T}
    options::Dict{Symbol, Dict{Symbol, Any}}
    atlas::A
    zeroproblem::Z
end

function ContinuationProblem(T::DataType=Float64)
    options = Dict{Symbol, Dict{Symbol, Any}}()
    atlas = Atlas(T)
    zeroproblem = ZeroProblem(T)
    return ContinuationProblem{T, Any, Any}(options, atlas, zeroproblem)
end

getatlas(prob::ContinuationProblem) = prob.atlas
getzeroproblem(prob::ContinuationProblem) = prob.zeroproblem

"""
    getoption(prob, toolbox, key; default=nothing)

Get the value of a user-supplied toolbox option.
"""
function getoption(prob::ContinuationProblem, toolbox::Symbol, key::Symbol; default=nothing)
    tbxoptions = get(prob.options, toolbox, nothing)
    return tbxoptions === nothing ? default : get(tbxoptions, key, default)
end

"""
    setoption!(prob, toolbox, key, value)

Set a toolbox option with a user-supplied value.
"""
function setoption!(prob::ContinuationProblem, toolbox::Symbol, key::Symbol, value)
    if !(toolbox in keys(prob.options))
        prob.options[toolbox] = Dict{Symbol, Any}()
    end
    prob.options[toolbox][key] = value
    return prob
end

function specialize(prob::ContinuationProblem{T}) where T
    atlas = specialize(prob.atlas)
    zeroproblem = specialize(prob.zeroproblem)
    return ContinuationProblem{T}(prob.options, atlas, zeroproblem)
end