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
using ..ZeroProblems: Var, AbstractZeroProblem, ParameterFunction, addparameter, 
    nextproblemname
import ..ZeroProblems: residual!

export AlgebraicProblem, AlgebraicProblem!

struct AlgebraicZeroProblem{T, F, U, P} <: AbstractZeroProblem{T}
    name::Symbol
    deps::Vector{Var{T}}
    f!::F
    fdim::Int64
    vars::Dict{Symbol, Var{T}}
end

"""
    AlgebraicZeroProblem(f, u0, p0; name)
"""
function AlgebraicZeroProblem(f, u0::Union{Number, Vector{<: Number}}, p0::Union{Number, Vector{<: Number}}; name)
    # Determine whether f is in-place or not
    if any(method.nargs == 4 for method in methods(f))
        f! = f
    else
        f! = (res, u, p) -> res .= f(u, p)
    end
    # Construct the continuation variables
    u = Var(:u, length(u0), u0=u0)
    p = Var(:p, length(p0), u0=p0)
    # Helpers
    T = numtype(u)
    U = u0 isa Vector ? Vector{T} : T
    P = p0 isa Vector ? Vector{T} : T
    AlgebraicZeroProblem{T, typeof(f!), U, P}(name, [u, p], f!, length(u0), Dict(:u=>u, :p=>p))
end

_convertto(T, val) = val
_convertto(::Type{T}, val) where {T <: Number} = val[1]

residual!(res, ap::AlgebraicZeroProblem{<: Any, <: Any, U, P}, u, p) where {U, P} = ap.f!(res, _convertto(U, u), _convertto(P, p))

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
    efunc::AlgebraicZeroProblem{T}
    mfuncs::Vector{ParameterFunction{T}}
end

function AlgebraicProblem(f, u0::Union{Number, Vector{<: Number}}, p0::Union{Number, Vector{<: Number}}; pnames=(), name=:alg)
    if !isempty(pnames) && (length(p0) !== length(pnames))
        throw(ArgumentError("p0 and pnames are not the same length ($(length(p0)) and $(length(pnames)) respectively)"))
    end
    efunc = AlgebraicZeroProblem(f, u0, p0, name=name)
    T = numtype(efunc)
    p = efunc[:p]  # parameter vector
    mfuncs = Vector{ParameterFunction{T}}()
    _pnames = isempty(pnames) ? map(i -> Symbol(:p, i), 1:length(p0)) : pnames
    for (i, pname) in pairs(_pnames)
        push!(mfuncs, addparameter(Var(pname, 1, parent=p, offset=i-1), name=Symbol(name, :_mfunc, i)))
    end
    return AlgebraicProblem{T}(name, efunc, mfuncs)
end

AlgebraicProblem!(prob::AbstractContinuationProblem, args...; name=:alg, kwargs...) = 
    push!(prob, AlgebraicProblem(args...; name=nextproblemname(prob, name), kwargs...))

getsubproblems(alg::AlgebraicProblem) = [alg.efunc; alg.mfuncs]

end # module
