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
coll = Var("coll", 20)  # an array of 20 Float64 (or other numerical type)
x0 = Var("x0", 1, coll, 0)  # a single Float64 at the start of the collocation array
x1 = Var("x1", 1, coll, -1)  # a single Float64 at the end of the collocation array
```
"""
mutable struct Var
    name::String
    len::Int64
    parent::Union{Var, Nothing}
    offset::Int64
    Var(name::String, len::Int64, parent::Var, offset::Int64) = new(nameof(parent)*"."*name, len, parent, offset)
    Var(name::String, len::Int64) = new(name, len, nothing, 0)
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
struct ZeroSubproblem{U, F} <: AbstractZeroSubproblem
    name::String
    deps::Vector{Var}
    f!::F
    fdim::Int64
    u0::U  # initial value
    t0::U  # initial tangent
    correct::Bool  # whether the corrector should be applied to the initial solution
end

"""
    ZeroSubproblem(f; fdim, u0, t0=nothing, correct=true, name="zero")
"""
function ZeroSubproblem(f; fdim=0, u0, t0=nothing, correct=true, name="zero")
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
    u = Var("u", length(u0))
    # Initial tangent
    if t0 === nothing
        _t0 = zero(u0)
    else
        if length(t0) != length(u0)
            throw(ArgumentError("Initial tangent should be the same size as the initial solution"))
        end
        _t0 = copy(t0)
    end
    return ZeroSubproblem(name, [u], f!, fdim, copy(u0), _t0, correct)
end

residual!(res, zp::ZeroSubproblem, u) = zp.f!(res, u)
fdim(zp::ZeroSubproblem) = zp.fdim
getinitial(zp::ZeroSubproblem) = (u=zp.u0, TS=zp.t0, data=nothing, correct=zp.correct)

#--- ZeroProblem - the full problem structure

struct ZeroProblem{D, U, Φ}
    u::U
    ui::Vector{UnitRange{Int64}}
    ϕ::Φ
    ϕi::Vector{UnitRange{Int64}}
    ϕdeps::Vector{Tuple{Vararg{Int64, N} where N}}
end

ZeroProblem() = ZeroProblem{Nothing, Vector{Var}, Vector{Any}}(Vector{Var}(), Vector{UnitRange{Int64}}(), Vector{Any}(), Vector{UnitRange{Int64}}(), Vector{Tuple{Vararg{Int64, N} where N}}())

function Base.push!(zp::ZeroProblem{Nothing}, u::Var)
    if !(u in zp.u)
        if u.parent === nothing
            last = isempty(zp.ui) ? 0 : maximum(maximum.(zp.ui))
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

function Base.push!(zp::ZeroProblem{Nothing}, subprob::AbstractZeroSubproblem)
    if subprob in zp.ϕ
        throw(ArgumentError("Subproblem is already part of the zero problem"))
    end
    depidx = Vector{Int64}()
    for dep in dependencies(subprob)
        push!(zp, dep)
        push!(depidx, findfirst(isequal(dep), zp.u))
    end
    last = isempty(zp.ϕi) ? 0 : maximum(maximum.(zp.ϕi))
    push!(zp.ϕ, subprob)
    push!(zp.ϕi, last + 1:last + fdim(subprob))
    push!(zp.ϕdeps, (depidx...,))
    return zp
end

function specialize(zp::ZeroProblem)
    u = (zp.u...,)
    ui = zp.ui
    ϕ = (zp.ϕ...,)
    ϕi = zp.ϕi
    ϕdeps = zp.ϕdeps
    return ZeroProblem{(ϕdeps...,), typeof(u), typeof(ϕ)}(u, ui, ϕ, ϕi, ϕdeps)
end

function residual!(res, zp::ZeroProblem{Nothing}, u, prob)
    # TODO: implement anyway?
    throw(ArgumentError("Specialize the zero problem before calling residual!"))
end

@generated function residual!(res, zp::ZeroProblem{D, U, Φ}, u, prob) where {D, U <: Tuple, Φ <: Tuple}
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
