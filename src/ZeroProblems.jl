module ZeroProblems

using UnsafeArrays

#--- Exports

export ZeroProblem, ZeroSubproblem, Var, residual!, fdim, udim

#--- Forward definitions

"""
    nameof(u)

Return the name of a variable or zero problem as a String.
"""
function nameof end

"""
    dependencies(z)

Return the variable dependencies of a zero problem as a Tuple.
"""
function dependencies end

"""
    passproblem(z)

A trait to determine whether the full problem structure is passed down to a
particular subtype of AbstractZeroSubproblem. The default is false.

# Example

A ZeroSubproblem containing the pseudo-arclength equation might require the
current tangent vector (which can be extracted from the problem structure) and
so it defines

```
passproblem(z::Type{PseudoArclength}) = true
```
"""
passproblem(z) = false

"""
    residual!(res, [J], z, u, [prob])

Return the residual (inplace), and optionally the Jacobian, of the ZeroProblem
z with the input u. Some ZeroProblems also require the problem structure
`prob` to be passed.
"""
function residual! end

"""
    fdim(f)

Return the dimension of functions contained within `f`.
"""
function fdim end

"""
    udim(u)

Return the dimension of the variables contained within `u`.
"""
function udim end

"""
    getinitial(prob)

Return the initial data (solution, tangent, toolbox data, correct) used for
initialising the continuation.
"""
function getinitial end

#--- Variables that zero problems depend on

"""
    Var

A placeholder for a continuation variable. As a minimum it comprises a name
and length (in units of the underlying numerical type). Optionally it can
include a parent variable and offset into the parent variable (negative
offsets represent the offset from the end). If a parent is specified, the name
is prefixed with the parent's name.

# Example

```
coll = Var(Float64, "coll", 20)  # an array of 20 Float64
x0 = Var(Float64, "x0", 1, coll, 0)  # a single Float64 at the start of the collocation array
x1 = Var(Float64, "x1", 1, coll, -1)  # a single Float64 at the end of the collocation array
```
"""
mutable struct Var{T}
    name::String
    len::Int64
    u0::Vector{T}
    t0::Vector{T}
    parent::Union{Var{T}, Nothing}
    offset::Int64
end

_convertvec(T, val::Nothing, len) = zeros(T, len)
_convertvec(T, val::Number, len) = T[val]
_convertvec(T, val::Vector, len) = convert(Vector{T}, val) 

function Var(T, name::String, len::Int64; parent::Union{Var, Nothing}=nothing, offset=0, u0=nothing, t0=nothing)
    if parent !== nothing
        if (u0 !== nothing) || (t0 !== nothing)
            throw(ArgumentError("Cannot have both a parent and u0 and/or t0"))
        end
        name = nameof(parent)*"."*name
    end
    _u0 = _convertvec(T, u0, len)
    _t0 = _convertvec(T, t0, len)
    if !(length(_u0) == length(_t0) == len)
        throw(ArgumentError("u0 and/or t0 must be nothing or Vectors of length len"))
    end
    return Var{T}(name, len, _u0, _t0, parent, offset)
end

nameof(u::Var) = u.name
udim(u::Var) = u.len

#--- ZeroSubproblem

# For simple subproblems the ZeroSubproblem type can be used to define a
# subproblem and its associated variable dependencies.

# For more complicated subproblems (or subproblems that require access to the
# full problem structure) you should inherit from AbstractZeroSubproblem.

abstract type AbstractZeroSubproblem end

nameof(subprob::AbstractZeroSubproblem) = subprob.name
dependencies(subprob::AbstractZeroSubproblem) = subprob.deps
udim(subprob::AbstractZeroSubproblem) = sum(udim(dep) for dep in dependencies(subprob))

"""
    ZeroSubproblem <: AbstractZeroSubproblem

A lightweight wrapper around a single-argument vector-valued function for
convenience. Both in-place and not in-place functions are handled.
"""
struct ZeroSubproblem{T, F} <: AbstractZeroSubproblem
    name::String
    deps::Vector{Var{T}}
    f!::F
    fdim::Int64
end

"""
    ZeroSubproblem(f; fdim, u0, t0=nothing, correct=true, name="zero")
"""
function ZeroSubproblem(f, u0::Vector; fdim=0, t0=nothing, name="zero")
    # Determine whether f is in-place or not
    if any(method.nargs == 3 for method in methods(f))
        f! = f
        if fdim == 0
            throw(ArgumentError("For in-place functions the number of dimensions (fdim) must be specified"))
        end
    else
        f! = (res, u) -> res .= f(u)
        if fdim == 0
            res = f(u0)
            fdim = length(res)
        end
    end
    # Construct the continuation variables
    T = t0 === nothing ? eltype(u0) : promote_type(eltype(u0), eltype(t0))
    u = Var(T, name*".u", length(u0), u0=u0, t0=t0)
    return ZeroSubproblem(name, [u], f!, fdim)
end

residual!(res, zp::ZeroSubproblem, u) = zp.f!(res, u)
fdim(zp::ZeroSubproblem) = zp.fdim
getinitial(zp::ZeroSubproblem) = (data=nothing)

#--- ZeroProblem - the full problem structure

struct ZeroProblem{T, D, U, Φ}
    u::U
    ui::Vector{UnitRange{Int64}}
    ϕ::Φ
    ϕi::Vector{UnitRange{Int64}}
    ϕdeps::Vector{Tuple{Vararg{Int64, N} where N}}
end

ZeroProblem(T=Float64) = ZeroProblem{T, Nothing, Vector{Var}, Vector{Any}}(Vector{Var}(), Vector{UnitRange{Int64}}(), Vector{Any}(), Vector{UnitRange{Int64}}(), Vector{Tuple{Vararg{Int64, N} where N}}())

udim(zp::ZeroProblem) = isempty(zp.ui) ? 0 : maximum(maximum.(zp.ui))
fdim(zp::ZeroProblem) = isempty(zp.ϕi) ? 0 : maximum(maximum.(zp.ϕi))

function Base.push!(zp::ZeroProblem{T, Nothing}, u::Var) where T
    if !(u in zp.u)
        if u.parent === nothing
            last = udim(zp)
            push!(zp.u, u)
            push!(zp.ui, last + 1:last + udim(u))
        else
            idx = findfirst(isequal(u.parent), zp.u)
            start = (u.offset < 0) ? (zp.ui[idx][end] + u.offset + 1) : (zp.ui[idx][1] + u.offset)
            push!(zp.u, u)
            push!(zp.ui, start:start + udim(u) - 1)
        end
    end
    return zp
end

function Base.push!(zp::ZeroProblem{T, Nothing}, subprob::AbstractZeroSubproblem) where T
    if subprob in zp.ϕ
        throw(ArgumentError("Subproblem is already part of the zero problem"))
    end
    depidx = Vector{Int64}()
    for dep in dependencies(subprob)
        push!(zp, dep)
        push!(depidx, findfirst(isequal(dep), zp.u))
    end
    last = fdim(zp)
    push!(zp.ϕ, subprob)
    push!(zp.ϕi, last + 1:last + fdim(subprob))
    push!(zp.ϕdeps, (depidx...,))
    return zp
end

function specialize(zp::ZeroProblem{T}) where T
    u = (zp.u...,)
    ui = zp.ui
    ϕ = (zp.ϕ...,)
    ϕi = zp.ϕi
    ϕdeps = zp.ϕdeps
    return ZeroProblem{T, (ϕdeps...,), typeof(u), typeof(ϕ)}(u, ui, ϕ, ϕi, ϕdeps)
end

function residual!(res, zp::ZeroProblem{T, Nothing}, u, prob) where T
    # TODO: implement anyway?
    throw(ArgumentError("Specialize the zero problem before calling residual!"))
end

@generated function residual!(res, zp::ZeroProblem{T, D, U, Φ}, u, prob) where {T, D, U <: Tuple, Φ <: Tuple}
    body = quote
        # Construct views into u for each variable
        uv = ($((:(uview(u, zp.ui[$i])) for i in eachindex(U.parameters))...),)
        # Construct views into res for each subproblem
        resv = ($((:(uview(res, zp.ϕi[$i])) for i in eachindex(Φ.parameters))...),)
    end
    # Call each of the subproblems
    for i in eachindex(D)
        if length(D[i]) == 0
            # No dependencies means pass everything
            expr = :(residual!(resv[$i], zp.ϕ[$i], u))
        else
            expr = :(residual!(resv[$i], zp.ϕ[$i], $((:(uv[$(D[i][j])]) for j in eachindex(D[i]))...)))
        end
        if passproblem(Φ.parameters[i])
            push!(expr.args, :prob)
        end
        push!(body.args, expr)
    end
    # Return res
    push!(body.args, :res)
    # @show body
    body
end

end
