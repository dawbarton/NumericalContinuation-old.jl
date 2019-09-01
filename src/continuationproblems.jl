"""
    AbstractContinuationProblem

Super-type of all continuation problems.

# Interface

All subtypes are expected to implement:

* `numtype`
* `getatlas`
* `getzeroproblem`
* `gettoolboxes`
* `getoption`
* `setoption!`
* `add!`
* `solve!`
"""
AbstractContinuationProblem

# (prob::T)(args...; kwargs...) where T <: AbstractContinuationProblem = solve!(prob, args...; kwargs...)

mutable struct ContinuationProblem{T} <: AbstractContinuationProblem
    solved::Bool
    options::Dict{Symbol, Dict{Symbol, Any}}
    atlas::Any
    zp::ExtendedZeroProblem{T}
    toolboxes::Vector{AbstractToolbox{T}}
end

function ContinuationProblem(T::Type=Float64)
    solved = false
    options = Dict{Symbol, Dict{Symbol, Any}}()
    atlas = Atlas
    zp = ExtendedZeroProblem(T)
    return ContinuationProblem{T}(solved, options, atlas, zp, AbstractToolbox{T}[])
end

function specialize!(prob::ContinuationProblem{T}) where T
    prob.atlas = specialize(prob.atlas)
    prob.zp = specialize(prob.zp)
    for i in eachindex(prob.toolboxes)
        prob.toolboxes[i] = specialize(prob.toolboxes[i])
    end
end

function Base.show(io::IO, prob::ContinuationProblem{T}) where T
    print(io, "ContinuationProblem")
    (T !== Float64) && print(io, "{$T}")
    print(io, "(Dict(")
    print(io, join([":$(key) => $(length(prob.options[key])) options set" for key in keys(prob.options)], ", "))
    print(io, "), ")
    print(io, prob.zp)
    print(io, ", AbstractToolbox[")
    print(io, join([string(tbx) for tbx in prob.toolboxes], ", "))
    print(io, "])")
end

Base.getindex(prob::ContinuationProblem, x) = getindex(prob.zp, x)

numtype(::ContinuationProblem{T}) where T = T

getatlas(prob::ContinuationProblem) = prob.atlas  # TODO: check for usage since it will return ::Any
getzeroproblem(prob::ContinuationProblem) = prob.zp
gettoolboxes(prob::ContinuationProblem) = prob.toolboxes

function add!(prob::ContinuationProblem, tbx::AbstractToolbox)
    for subprob in getsubproblems(tbx)
        addfunc!(prob, subprob)
    end
    push!(prob.toolboxes, tbx)
    return tbx
end

add!(prob::ContinuationProblem, u::Var) = addvar!(prob.zp, u)
add!(prob::ContinuationProblem, f::ComputedFunction) = addfunc!(prob.zp, f)

"""
    getoption(prob, toolbox, key; default=nothing)

Get the value of a user-supplied toolbox option.
"""
function getoption(prob::ContinuationProblem, toolbox::Symbol, key::Symbol; default, T=typeof(default))
    tbxoptions = get(prob.options, toolbox, nothing)
    return tbxoptions === nothing ? default : convert(T, get(tbxoptions, key, default))
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
    solve!(prob, [contpars...])

Solve the continuation problem as `contpars` are varied.

# Example

```
prob = ContinuationProblem()
push!(prob, AlgebraicProblem((u, p) -> u^3 - u - p, 1, 0, pnames=[:p])))
continuation(prob, :p => (-1, +1))
```
"""
function solve! end

function solve!(prob::ContinuationProblem{T}, contpars::Union{Tuple, AbstractVector}) where T
    prob.solved && throw(ErrorException("Problem has previously been used with solve!; recreate the problem to avoid unexpected results"))
    prob.solved = true
    # Set up the continuation variables
    vars = Var{T}[]
    for contpar in contpars
        if contpar isa Pair
            # Expect to be of the form :p => (min_p, max_p)
            u = contpar[1] isa Var ? contpar[1] : getvar(prob, contpar[1])
            u₋ = contpar[2][1]
            u₊ = contpar[2][2]
            # addevent!(prob, u, u₋, Events.EP)  # TODO
            # addevent!(prob, u, u₊, Events.EP)  # TODO
        else
            u = contpar isa Var ? contpar : getvar(prob, contpar)
        end
        push!(vars, u)
        setvaractive!(prob.zp, u, true)
    end
    # Construct the atlas if necessary
    if prob.atlas isa Type
        prob.atlas = prob.atlas(prob, vars[1])  # the first continuation variable is the primary one
    end
    # Specialize as necessary
    specialize!(prob)
    # Do the continuation
    return prob.atlas(prob)
end

solve!(prob::ContinuationProblem, contpars...) = solve!(prob, contpars)
