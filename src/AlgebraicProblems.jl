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
import ..ZeroProblems: residual!, fdim, getinitial

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
ap = AlgebraicProblem((u, p) -> u^3 - p, 1.5, 1)  # u0 = 1.5, p0 = 1
push!(prob, ap)
```
"""
struct AlgebraicProblem{T, F, U, P} <: AbstractZeroSubproblem
    name::String
    deps::Vector{Var{T}}
    f!::F
    fdim::Int64
end

"""
    AlgebraicProblem(f, u0, p0; pnames=nothing, name="alg")
"""
function AlgebraicProblem(f, u0, p0; pnames::Vector=[], name="alg")
    if !isempty(pnames) && (length(p0) !== length(pnames))
        throw(ArgumentError("p0 and pnames are not the same length ($(length(p0)) and $(length(pnames)) respectively)"))
    end
    # Determine whether f is in-place or not
    if any(method.nargs == 4 for method in methods(f))
        f! = f
    else
        f! = (res, u, p) -> res .= f(u, p)
    end
    T = promote_type(eltype(u0), eltype(p0))
    U = u0 isa Vector ? Vector{T} : T
    P = p0 isa Vector ? Vector{T} : T
    # Construct the continuation variables
    u = Var(T, name*".u", length(u0), u0=u0)
    p = Var(T, name*".p", length(p0), u0=p0)
    # TODO: add monitor functions for the parameters (cf. coco_add_pars)
    AlgebraicProblem{T, typeof(f!), U, P}(name, [u, p], f!, length(u0))
end

function Base.show(io::IO, ap::AlgebraicProblem)
    n = fdim(ap)
    eqn = n == 1 ? "1 equation" : "$n equations"
    write(io, "AlgebraicProblem with $eqn")
end

Base.eltype(::AlgebraicProblem{T}) where T = T

_convertto(T, val) = val
_convertto(::Type{T}, val) where {T <: Number} = val[1]

residual!(res, ap::AlgebraicProblem{<: Any, <: Any, U, P}, u, p) where {U, P} = ap.f!(res, _convertto(U, u), _convertto(P, p))
fdim(ap::AlgebraicProblem) = ap.fdim
getinitial(ap::AlgebraicProblem) = (data=nothing)

end # module
