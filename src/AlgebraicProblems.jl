"""
    module AlgebraicProblems

This module implements basic functionality to continue algebraic problems of the
form

```math
    0 = f(u, p)
```
"""
module AlgebraicProblems

using ..NumericalContinuation: numtype, AbstractToolbox, AbstractContinuationProblem
import ..NumericalContinuation: getsubproblems
using ..ZeroProblems: Var, ZeroProblem, addparameter, nextproblemname
import ..ZeroProblems: residual!

export AlgebraicProblem, AlgebraicProblem!

struct AlgebraicZeroProblem{F, U, P}
    f!::F
end

"""
    AlgebraicZeroProblem(f, u0, p0; name)
"""
function algebraiczeroproblem(f, u0::Union{Number, Vector{<: Number}}, p0::Union{Number, Vector{<: Number}}; name)
    # Determine whether f is in-place or not
    if any(method.nargs == 4 for method in methods(f))
        f! = f
    else
        f! = (res, u, p) -> res .= f(u, p)
    end
    # Construct the continuation variables
    u = Var(Symbol(name, "_", :u), length(u0), u0=u0)
    p = Var(Symbol(name, "_", :p), length(p0), u0=p0)
    # Helpers
    T = numtype(u)
    U = u0 isa Vector ? Vector{T} : T
    P = p0 isa Vector ? Vector{T} : T
    alg = AlgebraicZeroProblem{typeof(f!), U, P}(f!)
    return ZeroProblem(alg, (u, p), fdim=length(u0), inplace=true, name=name)
end

_convertto(T, val) = val
_convertto(::Type{T}, val) where {T <: Number} = val[1]

residual!(res, ap::AlgebraicZeroProblem{<:Any, U, P}, u, p) where {U, P} = ap.f!(res, _convertto(U, u), _convertto(P, p))

"""
    AlgebraicProblem <: AbstractToolbox

A wrapper around an algebraic zero problem of the form

```math
    0 = f(u, p)
```

The function can operate on scalars or vectors, and be in-place or not. It
assumes that the function output is of the same dimension as `u`.   

# Example

```
using NumericalContinuation
using NumericalContinuation.AlgebraicProblems
prob = ContinuationProblem()
ap = AlgebraicProblem((u, p) -> u^3 - p, 1.5, 1)  # u0 = 1.5, p0 = 1
push!(prob, ap)
```
"""
struct AlgebraicProblem{T} <: AbstractToolbox{T}
    name::Symbol
    efunc::ZeroProblem{T}
    mfuncs::Vector{ZeroProblem{T}}
end

function AlgebraicProblem(f, u0::Union{Number, Vector{<: Number}}, p0::Union{Number, Vector{<: Number}}; pnames=nothing, name=:alg)
    if (pnames !== nothing) && (length(p0) !== length(pnames))
        throw(ArgumentError("p0 and pnames are not the same length ($(length(p0)) and $(length(pnames)) respectively)"))
    end
    efunc = algebraiczeroproblem(f, u0, p0, name=name)
    T = numtype(efunc)
    p = efunc[2]  # parameter vector
    mfuncs = Vector{ZeroProblem{T}}()
    _pnames = pnames === nothing ? [Symbol(name, :_p, i) for i in 1:length(p0)] : pnames
    for i in eachindex(_pnames)
        push!(mfuncs, addparameter(Var(Symbol(""), 1, parent=p, offset=i-1), name=_pnames[i]))
    end
    return AlgebraicProblem{T}(name, efunc, mfuncs)
end

function AlgebraicProblem!(prob::AbstractContinuationProblem, args...; name=:alg, kwargs...)
    subprob = AlgebraicProblem(args...; name=nextproblemname(prob, name), kwargs...)
    push!(prob, subprob)
    return subprob
end

getsubproblems(alg::AlgebraicProblem) = [alg.efunc; alg.mfuncs]

end # module
