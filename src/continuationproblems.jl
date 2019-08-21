"""
    AbstractContinuationProblem{T}

Super-type of all continuation problems, where `T` is the underlying numerical
type.

# Interface

All subtypes are expected to implement:

* `getatlas`
* `getzeroproblem`
* `gettoolboxes`
* `getoption`
* `setoption!`
* `continuation!`
"""
AbstractContinuationProblem

# (prob::T)(args...; kwargs...) where T <: AbstractContinuationProblem = continuation!(prob, args...; kwargs...)

mutable struct ContinuationProblem{T} <: AbstractContinuationProblem{T}
    options::Dict{Symbol, Dict{Symbol, Any}}
    atlas::AbstractAtlas{T}
    zeroproblem::ExtendedZeroProblem{T}
    toolboxes::Vector{AbstractToolbox{T}}
    contpars::Vector{Var{T}}
end

function ContinuationProblem(T::Type=Float64)
    options = Dict{Symbol, Dict{Symbol, Any}}()
    atlas = Atlas(T)
    zeroproblem = ExtendedZeroProblem(T)
    return ContinuationProblem{T}(options, atlas, zeroproblem, AbstractToolbox{T}[], Var{T}[])
end

function specialize(prob::ContinuationProblem{T}) where T
    atlas = specialize(prob.atlas)
    zeroproblem = specialize(prob.zeroproblem)
    toolboxes = AbstractToolbox{T}[specialize(tbx) for tbx in prob.toolboxes]
    return ContinuationProblem{T}(prob.options, atlas, zeroproblem, toolboxes, prob.contpars)
end

getatlas(prob::ContinuationProblem) = prob.atlas
getzeroproblem(prob::ContinuationProblem) = prob.zeroproblem
gettoolboxes(prob::ContinuationProblem) = prob.toolboxes

function Base.push!(prob::ContinuationProblem, tbx::AbstractToolbox)
    for subprob in getsubproblems(tbx)
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

"""
    continuation(prob, [contpars...])

Solve the continuation problem as `contpars` are varied.

# Example

```
prob = ContinuationProblem()
push!(prob, AlgebraicProblem((u, p) -> u^3 - u - p, 1, 0, pnames=[:p])))
continuation(prob, :p => (-1, +1))
```
"""
function continuation(prob::ContinuationProblem, contpars...)
    for contpar in contpars
        if contpar isa Pair
            # Expect to be of the form :p => (min_p, max_p)
            u = contpar[1] isa Var ? contpar[1] : getvar(Var, prob, contpar[1])
            u₋ = contpar[2][1]
            u₊ = contpar[2][2]
            # addevent!(prob, u, u₋, Events.EP)
            # addevent!(prob, u, u₊, Events.EP)
        else
            u = contpar isa Var ? contpar : getvar(Var, prob, contpar)
        end
        push!(prob.contpars, u)
    end
    return Coverings.runstatemachine!(prob, prob.contpars)
end
