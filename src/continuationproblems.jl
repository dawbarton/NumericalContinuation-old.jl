#-------------------------------------------------------------------------------
mutable struct ContinuationProblem{T} <: AbstractContinuationProblem{T}
    options::Dict{Symbol, Dict{Symbol, Any}}
    atlas::AbstractAtlas{T}
    zeroproblem::ExtendedZeroProblem{T}
    toolboxes::Vector{AbstractToolbox{T}}
end

function ContinuationProblem(T::DataType=Float64)
    options = Dict{Symbol, Dict{Symbol, Any}}()
    atlas = Atlas(T)
    zeroproblem = ExtendedZeroProblem(T)
    return ContinuationProblem{T}(options, atlas, zeroproblem, AbstractToolbox{T}[])
end

getatlas(prob::ContinuationProblem) = prob.atlas
getzeroproblem(prob::ContinuationProblem) = prob.zeroproblem
gettoolboxes(prob::ContinuationProblem) = prob.toolboxes

function Base.push!(prob::ContinuationProblem, tbx::AbstractToolbox)
    for subprob in getzeroproblems(tbx)
        push!(prob, subprob)
    end
    push!(prob.toolboxes, tbx)
    return prob
end

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
    toolboxes = AbstractToolbox{T}[specialize(tbx) for tbx in prob.toolboxes]
    return ContinuationProblem{T}(prob.options, atlas, zeroproblem, toolboxes)
end
