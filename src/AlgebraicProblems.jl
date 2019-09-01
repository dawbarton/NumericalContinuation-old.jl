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
using ..ZeroProblems: Var, ComputedFunction, EmbeddedFunction, zeroproblem, parameter
import ..ZeroProblems: evaluate!

export AlgebraicProblem, AlgebraicProblem!

struct AlgebraicZeroProblem{F, U, P} <: EmbeddedFunction
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
    U = u0 isa Vector ? Vector : Number  # give the user provided function the input expected
    P = p0 isa Vector ? Vector : Number
    alg = AlgebraicZeroProblem{typeof(f!), U, P}(f!)
    return zeroproblem(alg, NamedTuple{(Symbol(name, "_", :u), Symbol(name, "_", :p))}((u0, p0)), fdim=length(u0), name=name)
end

_convertto(T, val) = val
_convertto(::Type{T}, val) where {T <: Number} = val[1]

evaluate!(res, ap::AlgebraicZeroProblem{<:Any, U, P}, u, p) where {U, P} = ap.f!(res, _convertto(U, u), _convertto(P, p))

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
    efunc::ComputedFunction{T}
    mfuncs::Vector{ComputedFunction{T}}
end

function AlgebraicProblem(f, u0::Union{Number, Vector{<: Number}}, p0::Union{Number, Vector{<: Number}}; pnames=nothing, name)
    if (pnames !== nothing) && (length(p0) !== length(pnames))
        throw(ArgumentError("p0 and pnames are not the same length ($(length(p0)) and $(length(pnames)) respectively)"))
    end
    efunc = algebraiczeroproblem(f, u0, p0, name=name)
    T = numtype(efunc)
    p = efunc[2]  # parameter vector
    mfuncs = Vector{ComputedFunction{T}}()
    _pnames = pnames === nothing ? [Symbol(name, :_p, i) for i in 1:length(p0)] : pnames
    for i in eachindex(_pnames)
        push!(mfuncs, parameter(Var(Symbol(""), 1, parent=p, offset=i-1), name=_pnames[i]))
    end
    return AlgebraicProblem{T}(name, efunc, mfuncs)
end

getsubproblems(alg::AlgebraicProblem) = [alg.efunc; alg.mfuncs]

end # module
