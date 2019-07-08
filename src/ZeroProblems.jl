module ZeroProblems

using UnsafeArrays
import ..NumericalContinuation: specialize

#--- Exports

export ZeroProblem, ZeroSubproblem, Var, residual!, fdim, udim, dependencies

#--- Forward definitions

"""
    dependencies(z)

Return the variable dependencies of a zero problem as a Tuple.
"""
function dependencies end

"""
    passproblem(z)

A trait to determine whether the full problem structure is passed down to a
particular subtype of AbstractZeroSubproblem. The default is false.

Also see `passdata`.

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
    passdata(z)

A trait to determine whether the function data is passed down to a particular
subtype of AbstractZeroSubproblem. The default is false.

Also see `passproblem`.

# Example

A ZeroSubproblem containing collocation equations might require the time
discretization which it stores in its own data structure and so it defines

```
passdata(z::Type{Collocation}) = true
```
"""
passdata(z) = false

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

Return the initial data (solution, tangent, toolbox data) used for initialising
the continuation.
"""
function getinitial end

#--- Variables that zero problems depend on

"""
    Var

A placeholder for a continuation variable. As a minimum it comprises a name
and length. Optionally it can include a parent variable and offset into the
parent variable (negative offsets represent the offset from the end). If a
parent is specified, the name is prefixed with the parent's name.

# Example

```
coll = Var(:coll, 20, T=Float64)  # an array of 20 Float64
x0 = Var(:x0, 1, parent=coll, offset=0)  # a single Float64 at the start of the collocation array
x1 = Var(:x1, 1, parent=coll, offset=-1)  # a single Float64 at the end of the collocation array
x2 = Var(:x2, 4, u0=[1, 2, 3, 4])  # an array of Float64 (integers are auto-promoted) with initial values
```
"""
mutable struct Var{T}
    name::Symbol
    len::Int64
    u0::Vector{T}
    t0::Vector{T}
    parent::Union{Var{T}, Nothing}
    offset::Int64
end

_convertvec(T, val, len) = throw(ArgumentError("Expected a number or a vector of numbers as a continuation variable"))
_convertvec(T, val::Nothing, len) = zeros(T, len)
_convertvec(T, val::Number, len) = T[val]
_convertvec(T, val::Vector, len) = convert(Vector{T}, val) 

function Var(name::Symbol, len::Int64; parent::Union{Var, Nothing}=nothing, offset=0, u0=nothing, t0=nothing, T=nothing)
    if parent !== nothing
        if (u0 !== nothing) || (t0 !== nothing)
            throw(ArgumentError("Cannot have both a parent and u0 and/or t0"))
        end
        T = eltype(parent)
    else
        if T === nothing
            if u0 === nothing
                throw(ArgumentError("Either T or u0 must be specified"))
            end
            T = eltype(u0) <: Integer ? Float64 : eltype(u0)  # type promotion of integers since they don't make sense as continuation variables
        end
    end
    _u0 = _convertvec(T, u0, len)
    _t0 = _convertvec(T, t0, len)
    if !(length(_u0) == length(_t0) == len)
        throw(ArgumentError("u0 and/or t0 must be nothing or Vectors of length len"))
    end
    return Var{T}(name, len, _u0, _t0, parent, offset)
end

Base.nameof(u::Var) = u.name
udim(u::Var) = u.len
getinitial(u::Var) = (u=u.u0, TS=u.t0)
Base.eltype(u::Var{T}) where T = T
Base.parent(u::Var) = u.parent

function Base.show(io::IO, u::Var{T}) where T
    varname = nameof(u)
    typename = nameof(T)
    n = udim(u)
    el = n == 1 ? "1 element" : "$n elements"
    print(io, "Variable ($varname) with $el ($typename)")
end

#--- ZeroSubproblem

# For simple subproblems the ZeroSubproblem type can be used to define a
# subproblem and its associated variable dependencies.

# For more complicated subproblems (or subproblems that require access to the
# full problem structure) you should inherit from AbstractZeroSubproblem.

abstract type AbstractZeroSubproblem{T} end

Base.nameof(subprob::AbstractZeroSubproblem) = subprob.name
dependencies(subprob::AbstractZeroSubproblem) = subprob.deps
fdim(subprob::AbstractZeroSubproblem) = subprob.fdim
getinitial(subprob::AbstractZeroSubproblem) = (data=nothing,)
Base.eltype(subprob::AbstractZeroSubproblem{T}) where T = T
Base.getindex(subprob::AbstractZeroSubproblem, idx::Integer) = getindex(subprob.deps, idx)
Base.getindex(subprob::AbstractZeroSubproblem, sym::Symbol) = subprob.vars[sym]

function Base.show(io::IO, subprob::AbstractZeroSubproblem)
    typename = nameof(typeof(subprob))
    probname = nameof(subprob)
    neqn = fdim(subprob)
    eqn = neqn == 1 ? "1 dimension" : "$neqn dimensions"
    nvar = length(dependencies(subprob))
    var = nvar == 1 ? "1 variable" : "$nvar variables"
    print(io, "$typename ($probname) with $eqn and $var")
end

"""
    ZeroSubproblem <: AbstractZeroSubproblem

A lightweight wrapper around a single-argument vector-valued function for
convenience. Both in-place and not in-place functions are handled.
"""
struct ZeroSubproblem{T, F} <: AbstractZeroSubproblem{T}
    name::Symbol
    deps::Vector{Var{T}}
    f!::F
    fdim::Int64
    vars::Dict{Symbol, Var{T}}
end

function ZeroSubproblem(f, u0::Union{Tuple, NamedTuple}; fdim=0, t0=Iterators.repeated(nothing), name=:zero, inplace=false)
    # Construct continuation variables as necessary
    deps = Vector{Var}()  # abstract type - will specialize when constructing ZeroSubproblem
    for (u, t) in zip(pairs(u0), t0)
        if !(u[2] isa Var)
            varname = u[1] isa Symbol ? u[1] : Symbol(:u, u[1])
            if !((u[2] isa Number) || (u[2] isa Vector{<: Number}))
                throw(ArgumentError("Expected a number, a vector of numbers, or an existing continuation variable but got a $(typeof(u[2]))"))
            end
            push!(deps, Var(varname, length(u[2]), u0=u[2], t0=t))
        else
            push!(deps, u[2])
        end
    end
    T = eltype(first(deps))
    vars = Dict{Symbol, Var{T}}()
    for dep in deps
        vars[nameof(dep)] = dep
    end
    # Determine whether f is in-place or not
    if inplace
        f! = f
        if fdim == 0
            throw(ArgumentError("For in-place functions the number of dimensions (fdim) must be specified"))
        end
    else
        f! = (res, u...) -> res .= f(u...)
        if fdim == 0
            res = f((u.u0 for u in deps)...)
            fdim = length(res)
        end
    end
    # Construct the continuation variables
    return ZeroSubproblem(name, [deps...], f!, fdim, vars)
end
ZeroSubproblem(f, u0; t0=nothing, kwargs...) = ZeroSubproblem(f, (u0,); t0=(t0,), kwargs...)

residual!(res, zp::ZeroSubproblem, u...) = zp.f!(res, u...)

#--- ZeroProblem - the full problem structure

struct ZeroProblem{T, D, U, Φ}
    u::U
    ui::Vector{UnitRange{Int64}}
    ϕ::Φ
    ϕi::Vector{UnitRange{Int64}}
    ϕdeps::Vector{Tuple{Vararg{Int64, N} where N}}
    udim::Base.RefValue{Int64}
    ϕdim::Base.RefValue{Int64}
end

ZeroProblem(T=Float64) = ZeroProblem{T, Nothing, Vector{Var{T}}, Vector{AbstractZeroSubproblem{T}}}(Vector{Var{T}}(), Vector{UnitRange{Int64}}(), Vector{AbstractZeroSubproblem{T}}(), Vector{UnitRange{Int64}}(), Vector{Tuple{Vararg{Int64, N} where N}}(), Ref(0), Ref(0))

function ZeroProblem(subprobs::Vector{<: AbstractZeroSubproblem{T}}) where T
    zp = ZeroProblem(T)
    for subprob in subprobs
        push!(zp, subprob)
    end
    return zp
end

function specialize(zp::ZeroProblem{T}) where T
    u = (zp.u...,)
    ui = zp.ui
    ϕ = ((specialize(ϕ) for ϕ in zp.ϕ)...,)
    ϕi = zp.ϕi
    ϕdeps = zp.ϕdeps
    return ZeroProblem{T, (ϕdeps...,), typeof(u), typeof(ϕ)}(u, ui, ϕ, ϕi, ϕdeps, zp.udim, zp.ϕdim)
end

udim(zp::ZeroProblem) = zp.udim[]
fdim(zp::ZeroProblem) = zp.ϕdim[]

function update_ui(zp::ZeroProblem, u::Var, last)
    n = udim(u)
    if u.parent === nothing
        ui = (last + 1):(last + n)
        last += n
    else
        idx = findfirst(isequal(u.parent), zp.u)
        start = (u.offset < 0) ? (zp.ui[idx][end] + u.offset + 1) : (zp.ui[idx][1] + u.offset)
        ui = start:(start + n - 1)
    end
    return (ui, last)
end

function update_ui!(zp::ZeroProblem)
    last = 0
    for i in eachindex(zp.u)
        (ui, last) = update_ui(zp, zp.u[i], last)
        zp.ui[i] = ui
    end
    zp.udim[] = last
    return zp
end

function Base.push!(zp::ZeroProblem{T, Nothing}, u::Var) where T
    if !(u in zp.u)
        up = parent(u)
        if (up !== nothing) && !(up in zp.u)
            throw(ArgumentError("Parent variable is not contained in the zero problem"))
        end
        push!(zp.u, u)
        (ui, last) = update_ui(zp, u, zp.udim[])
        push!(zp.ui, ui)
        zp.udim[] = last
    end
    return zp
end

function update_ϕi!(zp::ZeroProblem)
    last = 0
    for i in eachindex(zp.ϕ)
        n = fdim(zp.ϕ[i])
        zp.ϕi[i] = (last + 1):(last + n)
        last += n
    end
    zp.ϕdim[] = last
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
    push!(zp.ϕ, subprob)
    last = zp.ϕdim[]
    ϕdim = fdim(subprob)
    push!(zp.ϕi, (last + 1):(last + ϕdim))
    zp.ϕdim[] = last + ϕdim
    push!(zp.ϕdeps, (depidx...,))
    return zp
end

function residual!(res, zp::ZeroProblem{T, Nothing}, u, prob=nothing, data=nothing) where T
    uv = [uview(u, zp.ui[i]) for i in eachindex(zp.ui)]
    for i in eachindex(zp.ϕ)
        args = Any[uview(res, zp.ϕi[i]), zp.ϕ[i]]
        if length(zp.ϕdeps[i]) == 0
            # No dependencies means pass everything
            push!(args, u)
        else
            for dep in zp.ϕdeps[i]
                push!(args, uv[dep])
            end
        end
        if (prob !== nothing) && passproblem(zp.ϕ[i])
            push!(args, prob)
        end
        if (data !== nothing) && passdata(zp.ϕ[i])
            push!(args, data[i])
        end
        residual!(args...)
    end
    return res
end

@generated function residual!(res, zp::ZeroProblem{T, D, U, Φ}, u, prob=nothing, data=nothing) where {T, D, U <: Tuple, Φ <: Tuple}
    body = quote
        # Construct views into u for each variable
        uv = ($((:(uview(u, zp.ui[$i])) for i in eachindex(U.parameters))...),)
    end
    # Call each of the subproblems
    for i in eachindex(D)
        if length(D[i]) == 0
            # No dependencies means pass everything
            expr = :(residual!(uview(res, zp.ϕi[$i]), zp.ϕ[$i], u))
        else
            expr = :(residual!(uview(res, zp.ϕi[$i]), zp.ϕ[$i], $((:(uv[$(D[i][j])]) for j in eachindex(D[i]))...)))
        end
        if (prob !== Nothing) && passproblem(Φ.parameters[i])
            push!(expr.args, :prob)
        end
        if (data !== Nothing) && passdata(ϕ.parameters[i])
            push!(expr.args, :(data[$i]))
        end
        push!(body.args, expr)
    end
    # Return res
    push!(body.args, :res)
    # @show body
    body
end

function getinitial(zp::ZeroProblem{T}) where T
    ndim = udim(zp)
    u = zeros(T, ndim)
    t = zeros(T, ndim)
    for i in eachindex(zp.u)
        if parent(zp.u[i]) === nothing
            u[zp.ui[i]] .= zp.u[i].u0
            t[zp.ui[i]] .= zp.u[i].t0
        end
    end
    data = ((getinitial(ϕ).data for ϕ in zp.ϕ)...,)
    return (u=u, TS=t, data=data)
end

end
