module ZeroProblems

using UnsafeArrays
using ..NumericalContinuation: AbstractContinuationProblem, getzeroproblem
import ..NumericalContinuation: specialize, numtype

using ForwardDiff

#--- Exports

export ExtendedZeroProblem, ZeroProblem, Var
export residual!, fdim, udim, fidx, uidx, dependencies

#--- Forward definitions

"""
    dependencies(z)

Return the variable dependencies of a zero problem as a Tuple.
"""
function dependencies end

"""
    passproblem(z)

A trait to determine whether the full problem structure is passed down to a
particular subtype of AbstractZeroProblem. The default is false.

Also see `passdata`.

# Example

A ZeroProblem containing the pseudo-arclength equation might require the
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
subtype of AbstractZeroProblem. The default is false.

Also see `passproblem`.

# Example

A ZeroProblem containing collocation equations might require the time
discretization which it stores in its own data structure and so it defines

```
passdata(z::Type{Collocation}) = true
```
"""
passdata(z) = false

"""
    residual!(res, [J], z, u, [prob])

Return the residual (inplace), and optionally the Jacobian, of the ExtendedZeroProblem
z with the input u. Some ExtendedZeroProblems also require the problem structure
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
        T = numtype(parent)
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
numtype(u::Var{T}) where T = T
Base.parent(u::Var) = u.parent

function Base.show(io::IO, u::Var{T}) where T
    varname = nameof(u)
    typename = nameof(T)
    n = udim(u)
    el = n == 1 ? "1 element" : "$n elements"
    print(io, "Variable ($varname) with $el ($typename)")
end

#--- Common helpers

function _constructdeps(u0, t0)
    # Construct continuation variables as necessary
    deps = Vector{Var}()  # abstract type - will specialize when returning
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
    T = numtype(first(deps))
    vars = Dict{Symbol, Var{T}}()
    for dep in deps
        if nameof(dep) in keys(vars)
            @warn "Duplicate variable name" dep
        end
        vars[nameof(dep)] = dep
    end
    return (convert(Vector{Var{T}}, deps), vars)
end

#--- ZeroProblem

# For simple problems the ZeroProblem type can be used to define a
# problem and its associated variable dependencies.

# For more complicated problems (or problems that require access to the
# full problem structure) you should inherit from AbstractZeroProblem.

abstract type AbstractZeroProblem{T} end

Base.nameof(prob::AbstractZeroProblem) = prob.name
dependencies(prob::AbstractZeroProblem) = prob.deps
fdim(prob::AbstractZeroProblem) = prob.fdim
getinitial(prob::AbstractZeroProblem) = (data=nothing,)
numtype(prob::AbstractZeroProblem{T}) where T = T
Base.getindex(prob::AbstractZeroProblem, idx::Integer) = getindex(prob.deps, idx)
Base.getindex(prob::AbstractZeroProblem, sym::Symbol) = prob.vars[sym]

function Base.show(io::IO, prob::AbstractZeroProblem)
    typename = nameof(typeof(prob))
    probname = nameof(prob)
    neqn = fdim(prob)
    eqn = neqn == 1 ? "1 dimension" : "$neqn dimensions"
    nvar = length(dependencies(prob))
    var = nvar == 1 ? "1 variable" : "$nvar variables"
    print(io, "$typename ($probname) with $eqn and $var")
end

"""
    ZeroProblem <: AbstractZeroProblem

A lightweight wrapper around a single-argument vector-valued function for
convenience. Both in-place and not in-place functions are handled.
"""
struct ZeroProblem{T, F} <: AbstractZeroProblem{T}
    name::Symbol
    deps::Vector{Var{T}}
    f!::F
    fdim::Int64
    vars::Dict{Symbol, Var{T}}
end

function ZeroProblem(f, u0::Union{Tuple, NamedTuple}; fdim=0, t0=Iterators.repeated(nothing), name=:zero, inplace=false)
    deps, vars = _constructdeps(u0, t0)
    # Determine whether f is in-place or not
    if inplace
        f! = f
        if fdim == 0
            throw(ArgumentError("For in-place functions the number of dimensions (fdim) must be specified"))
        end
    else
        f! = (res, u...) -> res .= f(u...)
        if fdim == 0
            res = f((getinitial(u).u for u in deps)...)
            fdim = length(res)
        end
    end
    # Construct the continuation variables
    return ZeroProblem(name, deps, f!, fdim, vars)
end
ZeroProblem(f, u0; t0=nothing, kwargs...) = ZeroProblem(f, (u0,); t0=(t0,), kwargs...)

residual!(res, zp::ZeroProblem, u...) = zp.f!(res, u...)

#--- MonitorFunction

mutable struct MonitorFunction{T, F} <: AbstractZeroProblem{T}
    name::Symbol
    deps::Vector{Var{T}}
    f::F
    vars::Dict{Symbol, Var{T}}
    active::Bool
end

function MonitorFunction(f, u0::Union{Tuple, NamedTuple}; t0=Iterators.repeated(nothing), name=:mfunc, active=false)
    # Construct continuation variables as necessary
    deps, vars = _constructdeps(u0, t0)
    udim = active ? 1 : 0
    μ = Var(name, udim, T=numtype(first(deps))) 
    insert!(deps, 1, μ)
    vars[name] = μ
    # Construct the continuation variables
    return MonitorFunction(name, deps, f, vars, active)
end
MonitorFunction(f, u0; t0=nothing, kwargs...) = MonitorFunction(f, (u0,); t0=(t0,), kwargs...)

fdim(mfunc::MonitorFunction) = 1
passdata(mfunc::MonitorFunction) = true
getinitial(mfunc::MonitorFunction) = (data=Ref(mfunc.f((getinitial(u).u for u in deps)...)),)

function residual!(res, mfunc::MonitorFunction, data, um, u...)
    μ = isempty(um) ? data[] : um[1]
    res[1] = mfunc.f(u...) - μ
end

#--- ExtendedZeroProblem - the full problem structure

struct ExtendedZeroProblem{T, D, U, Φ}
    u::U
    ui::Vector{UnitRange{Int64}}
    ϕ::Φ
    ϕi::Vector{UnitRange{Int64}}
    ϕdeps::Vector{Tuple{Vararg{Int64, N} where N}}
    udim::Base.RefValue{Int64}
    ϕdim::Base.RefValue{Int64}
end

ExtendedZeroProblem(T=Float64) = ExtendedZeroProblem{T, Nothing, Vector{Var{T}}, Vector{AbstractZeroProblem{T}}}(Vector{Var{T}}(), Vector{UnitRange{Int64}}(), Vector{AbstractZeroProblem{T}}(), Vector{UnitRange{Int64}}(), Vector{Tuple{Vararg{Int64, N} where N}}(), Ref(zero(Int64)), Ref(zero(Int64)))

function ExtendedZeroProblem(probs::Vector{<: AbstractZeroProblem{T}}) where T
    zp = ExtendedZeroProblem(T)
    for prob in probs
        push!(zp, prob)
    end
    return zp
end

function specialize(zp::ExtendedZeroProblem{T}) where T
    u = (zp.u...,)
    ui = zp.ui
    ϕ = ((specialize(ϕ) for ϕ in zp.ϕ)...,)
    ϕi = zp.ϕi
    ϕdeps = zp.ϕdeps
    return ExtendedZeroProblem{T, (ϕdeps...,), typeof(u), typeof(ϕ)}(u, ui, ϕ, ϕi, ϕdeps, zp.udim, zp.ϕdim)
end

"""
    udim(prob)

Return the number of independent continuation variables in the problem. (May
change during continuation, for example if adaptive meshing is used.)
"""
udim(zp::ExtendedZeroProblem) = zp.udim[]
udim(prob::AbstractContinuationProblem) = udim(getzeroproblem(prob))

"""
    fdim(prob)

Return the number of equations in the problem. (May change during continuation,
for example if adaptive meshing is used.)
"""
fdim(zp::ExtendedZeroProblem) = zp.ϕdim[]
fdim(prob::AbstractContinuationProblem) = fdim(getzeroproblem(prob))

"""
    uidx(prob, u::Var)

Return the index of the continuation variable within the internal structures.
This will not change during continuation and so can be stored for fast
indexing throughout the continuation run.

# Example

```
ui = uidx(prob, myvariable)  # once at the start of the continuation run (slow)
u[uidx(prob, ui)]  # as frequently as necessary (fast)
```
"""
uidx(zp::ExtendedZeroProblem, u::Var) = findfirst(isequal(u), zp.u)
uidx(prob::AbstractContinuationProblem, x) = uidx(getzeroproblem(prob), x)

"""
    uidx(prob, i::Integer)

Return the index of the continuation variable within the solution vector. (May
change during continuation, for example if adaptive meshing is used.)
"""
uidx(zp::ExtendedZeroProblem, i::Integer) = zp.ui[i]

"""
    fidx(prob, prob::AbstractZeroProblem)

Return the index of the sub-problem within the internal structures. This will
not change during continuation and so can be stored for fast indexing
throughout the continuation run.

# Example

```
fi = fidx(prob, myproblem)  # once at the start of the continuation run (slow)
res[fidx(prob, fi)]  # as frequently as necessary (fast)
```
"""
fidx(zp::ExtendedZeroProblem, prob::AbstractZeroProblem) = findfirst(isequal(prob), zp.ϕ)
fidx(prob::AbstractContinuationProblem, x) = fidx(getzeroproblem(prob), x)

"""
    fidx(prob, i::Integer)

Return the index of the sub-problem within the residual vector. (May change
during continuation, for example if adaptive meshing is used.)
"""
fidx(zp::ExtendedZeroProblem, i::Integer) = zp.ϕi[i]

function update_ui(zp::ExtendedZeroProblem, u::Var, last)
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

function update_ui!(zp::ExtendedZeroProblem)
    last = 0
    for i in eachindex(zp.u)
        (ui, last) = update_ui(zp, zp.u[i], last)
        zp.ui[i] = ui
    end
    zp.udim[] = last
    return zp
end

function Base.push!(zp::ExtendedZeroProblem{T, Nothing}, u::Var{T}) where T
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

Base.push!(prob::AbstractContinuationProblem{T}, u::Var{T}) where T = push!(getzeroproblem(prob), u)

function update_ϕi!(zp::ExtendedZeroProblem)
    last = 0
    for i in eachindex(zp.ϕ)
        n = fdim(zp.ϕ[i])
        zp.ϕi[i] = (last + 1):(last + n)
        last += n
    end
    zp.ϕdim[] = last
    return zp
end

function Base.push!(zp::ExtendedZeroProblem{T, Nothing}, prob::AbstractZeroProblem{T}) where T
    if prob in zp.ϕ
        throw(ArgumentError("Problem is already part of the zero problem"))
    end
    depidx = Vector{Int64}()
    for dep in dependencies(prob)
        push!(zp, dep)
        push!(depidx, findfirst(isequal(dep), zp.u))
    end
    push!(zp.ϕ, prob)
    last = zp.ϕdim[]
    ϕdim = fdim(prob)
    push!(zp.ϕi, (last + 1):(last + ϕdim))
    zp.ϕdim[] = last + ϕdim
    push!(zp.ϕdeps, (depidx...,))
    return zp
end

Base.push!(prob::AbstractContinuationProblem{T}, prob::AbstractZeroProblem{T}) where T = push!(getzeroproblem(prob), prob)

function residual!(res, zp::ExtendedZeroProblem{T, Nothing}, u, prob=nothing, data=nothing) where T
    uv = [uview(u, zp.ui[i]) for i in eachindex(zp.ui)]
    for i in eachindex(zp.ϕ)
        args = Any[uview(res, zp.ϕi[i]), zp.ϕ[i]]
        if (prob !== nothing) && passproblem(zp.ϕ[i])
            push!(args, prob)
        end
        if (data !== nothing) && passdata(zp.ϕ[i])
            push!(args, data[i])
        end
        if length(zp.ϕdeps[i]) == 0
            # No dependencies means pass everything
            push!(args, u)
        else
            for dep in zp.ϕdeps[i]
                push!(args, uv[dep])
            end
        end
        residual!(args...)
    end
    return res
end

@generated function residual!(res, zp::ExtendedZeroProblem{T, D, U, Φ}, u, prob=nothing, data=nothing) where {T, D, U <: Tuple, Φ <: Tuple}
    body = quote
        # Construct views into u for each variable
        uv = ($((:(uview(u, zp.ui[$i])) for i in eachindex(U.parameters))...),)
    end
    # Call each of the problems
    for i in eachindex(D)
        expr = :(residual!(uview(res, zp.ϕi[$i]), zp.ϕ[$i]))
        if (prob !== Nothing) && passproblem(Φ.parameters[i])
            push!(expr.args, :prob)
        end
        if (data !== Nothing) && passdata(Φ.parameters[i])
            push!(expr.args, :(data[$i]))
        end
        if length(D[i]) == 0
            # No dependencies means pass everything
            push!(expr.args, :u)
        else
            for j in eachindex(D[i])
                push!(expr.args, :(uv[$(D[i][j])]))
            end
        end
        push!(body.args, expr)
    end
    # Return res
    push!(body.args, :res)
    # @show body
    body
end

residual!(res, prob::AbstractContinuationProblem, u, args...) = residual!(res, getzeroproblem(prob), u, args...)

function jacobian!(J, zp::ExtendedZeroProblem{T}, u, args...) where T
    # A simple forward difference
    ϵ = T(1e-6)
    @assert size(J, 1) == size(J, 2) == length(u)
    res = zeros(T, length(u))
    residual!(res, zp, u, args...)
    for i in eachindex(u)
        uu = u[i]
        u[i] += ϵ
        residual!(uview(J, :, i), zp, u, args...)
        for j in eachindex(u)
            J[j, i] = (J[j, i] - res[j])/ϵ
        end
        u[i] = uu
    end
    return J
end

function jacobian_ad(zp::ExtendedZeroProblem, u, args...) 
    ForwardDiff.jacobian((res, u)->residual!(res, zp, u, args...), zeros(size(u)), u)
end

function getinitial(zp::ExtendedZeroProblem{T}) where T
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
