"""
    module AlgebraicProblems

This module implements basic functionality to continue algebraic problems of the
form

```math
    0 = f(u, p)
```
"""
module AlgebraicProblems

using ..ZeroProblems: Var, AbstractZeroSubproblem
import ..ZeroProblems: residual!, fdim

export AlgebraicProblem

"""
    AlgebraicProblem <: AbstractZeroSubproblem

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
ap = AlgebraicProblem((u, p) -> u^3 - p, u0=1.0, p0=1.0)
push!(prob, ap)
```
"""
struct AlgebraicProblem{U, P, F} <: AbstractZeroSubproblem
    name::String
    deps::Vector{Var}
    f!::F
    u0::U
    p0::P
end

"""
    AlgebraicProblem(f; u0, p0, pnames=nothing, name="alg")
"""
function AlgebraicProblem(f; u0, p0, pnames::Union{Vector, Tuple, Nothing}=nothing, name="alg")
    if (pnames !== nothing) && (length(p0) !== length(pnames))
        throw(ArgumentError("p0 and pnames are not the same length ($(length(p0)) and $(length(pnames)) respectively)"))
    end
    # Determine whether f is in-place or not
    if any(method.nargs == 4 for method in methods(f))
        f! = f
    else
        f! = (res, u, p) -> res .= f(u, p)
    end
    # Construct the continuation variables
    u = Var(name*".u", length(u0))
    p = Var(name*".p", length(p0))
    # TODO: add monitor functions for the parameters (cf. coco_add_pars)
    AlgebraicProblem(name, [u, p], f!, copy(u0), copy(p0))
end

convertto(T, val) = val
convertto(::Type{T}, val) where {T <: Number} = val[1]

residual!(res, ap::AlgebraicProblem{U, P}, u, p) where {U, P} = ap.f!(res, convertto(U, u), convertto(P, p))
fdim(ap::AlgebraicProblem) = length(ap.u0)

end # module
