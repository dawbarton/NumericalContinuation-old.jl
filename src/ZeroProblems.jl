module ZeroProblems

using UnsafeArrays: uview
using ..NumericalContinuation: AbstractContinuationProblem, getzeroproblem
import ..NumericalContinuation: specialize, numtype

import ForwardDiff

#--- Exports

export ExtendedZeroProblem, ComputedFunction, ComputedFunction!, Var, MonitorFunction,
    EmbeddedFunction, NonEmbeddedFunction
export evaluate!, evaluate_embedded!, evaluate_nonembedded!, fdim, udim, fidxrange, 
    uidxrange, dependencies, addparameter, addparameter!, getvar, getvars, 
    getfunc, getfuncs, hasvar, hasfunc, setvaractive!, isvaractive, zeroproblem,
    zeroproblem!, monitorfunction, monitorfunction!, addfunc!, addvar!

#--- Forward definitions

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
    idx::Int64
    idxrange::UnitRange{Int64}
end

_convertvec(T, val, len) = throw(ArgumentError("Expected a number or a vector of numbers as a continuation variable"))
_convertvec(T, val::Nothing, len) = zeros(T, len)
_convertvec(T, val::Number, len) = T[val]
_convertvec(T, val::Vector, len) = convert(Vector{T}, val) 

function Var(name::Symbol, len::Int64; parent::Union{Var, Nothing}=nothing, offset=0, u0=nothing, t0=nothing, T=nothing, idx=0, idxrange=0:0)
    if parent !== nothing
        if (u0 !== nothing) || (t0 !== nothing)
            throw(ArgumentError("Cannot have both a parent and u0 and/or t0"))
        end
        T = numtype(parent)
        # Copy u0 and t0 from the parent
        start = (offset < 0) ? (parent.len + offset + 1) : offset + 1
        _u0 = parent.u0[start:(start + len - 1)]
        _t0 = parent.t0[start:(start + len - 1)]
    else
        if T === nothing
            if u0 === nothing
                throw(ArgumentError("Either T or u0 must be specified"))
            end
            T = eltype(u0) <: Integer ? Float64 : eltype(u0)  # type promotion of integers since they don't make sense as continuation variables
        end
        _u0 = _convertvec(T, u0, len)
        _t0 = _convertvec(T, t0, len)
    end
    if !(length(_u0) == length(_t0) == len) && (len != 0)  # allow len = 0 for inactive monitor functions
        throw(ArgumentError("u0 and/or t0 must be nothing or Vectors of length len"))
    end
    return Var{T}(name, len, _u0, _t0, parent, offset, idx, idxrange)
end

Base.nameof(u::Var) = u.name
udim(u::Var) = u.len
initialvar(u::Var) = (u=u.u0, TS=u.t0)
numtype(u::Var{T}) where T = T
Base.parent(u::Var) = u.parent
uidxrange(u::Var) = u.idxrange

function Base.show(io::IO, u::Var{T}) where T
    _T = T === Float64 ? "" : "{$(nameof(T))}"
    name = nameof(u) === Symbol("") ? "UNNAMED" : ":$(nameof(u))"
    child = u.parent === nothing ? "" : ", parent=$(nameof(u.parent) === Symbol("") ? "UNNAMED" : ":$(nameof(u.parent))")"
    dim = ", $(u.len)"
    print(io, "Var$_T($name$dim$child)")
    return
end

#--- ComputedFunction

abstract type AbstractFunction end
abstract type EmbeddedFunction <: AbstractFunction end
abstract type NonEmbeddedFunction <: AbstractFunction end

"""
    ComputedFunction{T, F}

Note: not-inplace functions should not overload evaluate! but rely on being
called directly.
"""
mutable struct ComputedFunction{T, F <: AbstractFunction}
    name::Symbol
    deps::Vector{Var{T}}
    f!::F
    fdim::Int64
    vars::Dict{Symbol, Var{T}}
    idx::Int64
    idxrange::UnitRange{Int64}
end

function ComputedFunction(f!::AbstractFunction, u0::NTuple{N, Var{T}}; name, fdim) where {N, T}
    vars = Dict{Symbol, Var{T}}()
    for u in u0
        if nameof(u) in keys(vars)
            @warn "Duplicate variable name" u
        end
        vars[nameof(u)] = u
    end
    # Construct the continuation variables
    return ComputedFunction(name, [u for u in u0], f!, fdim, vars, 0, 0:0)
end

ComputedFunction(f!::AbstractFunction, u0::Var; kwargs...) = ComputedFunction(f!, (u0,); kwargs...)

function Base.show(io::IO, @nospecialize prob::ComputedFunction{T, F}) where {T, F}
    _T = T === Float64 ? "" : "$(nameof(T)), "
    name = nameof(prob) === Symbol("") ? "UNNAMED" : ":$(nameof(prob))"
    print(io, "ComputedFunction{$_T$(nameof(F))}($name, $(fdim(prob)))")
end

Base.nameof(prob::ComputedFunction) = prob.name
dependencies(prob::ComputedFunction) = prob.deps
fdim(prob::ComputedFunction) = prob.fdim
numtype(prob::ComputedFunction{T}) where T = T
Base.getindex(prob::ComputedFunction, idx::Integer) = getindex(prob.deps, idx)
Base.getindex(prob::ComputedFunction, sym::Symbol) = prob.vars[sym]
fidxrange(prob::ComputedFunction) = prob.idxrange
getfunc(prob::ComputedFunction) = prob.f!

evaluate!(res, f!::AbstractFunction, u...) = f!(res, u...)
evaluate!(res, prob::ComputedFunction, u...) = evaluate!(res, prob.f!, u...)

"""
    passproblem(::Type{MyProblem})

A trait to determine whether the full problem structure is passed down to a
particular `MyProblem <: AbstractFunction`. The default is `false`.

Also see `passdata`.

# Example

A ComputedFunction might require the current tangent vector (which can be
extracted from the problem structure) and so it defines

```
passproblem(::Type{MyProblem}) = true
```
"""
function passproblem end

passproblem(::Type{<: AbstractFunction}) = false
passproblem(::Type{<: ComputedFunction{T, F}}) where {T, F} = passproblem(F)

"""
    passdata(::Type{MyProblem})

A trait to determine whether the function data is passed down to a particular
`MyProblem <: AbstractFunction`. The default is `nothing`.

Also see `passproblem`.

# Example

A ComputedFunction containing collocation equations might require the time
discretization which it stores in its own data structure and so it defines

```
passdata(::Type{Collocation}) = true
```
"""
function passdata end

passdata(::Type{<: AbstractFunction}) = false
passdata(::Type{<: ComputedFunction{T, F}}) where {T, F} = passdata(F)


"""
    initialdata(prob)

Return the initial function data for the given function or function collection.

Also see `initialdata_embedded` and `initialdata_nonembedded`.
"""
function initialdata end

initialdata(::Any) = nothing  # cannot be restricted to AbstractFunctions because of MonitorFunction needing to delegate to a wider variety of 
initialdata(prob::ComputedFunction) = initialdata(prob.f!)

#--- ZeroProblem

"""
    ZeroProblem{F}

ZeroProblem is a lightweight wrapper for a function to signal that it is an
EmbeddedFunction. This is largely for user-defined functions for which
defining a full `struct` is unnecessary.

If options such as `passdata` or `passproblem` are desired, this wrapper
should not be used and a full `struct` (subtyped from `EmbeddedFunction`)
should be created.
"""
struct ZeroProblem{F} <: EmbeddedFunction
    f!::F
end

(zp::ZeroProblem)(u...) = zp.f!(u...)

function zeroproblem(f!, u0::Union{Tuple, NamedTuple}; t0=Iterators.repeated(nothing), name, kwargs...)
    # Construct continuation variables as necessary - zeroproblems can create new states
    deps = Vector{Var}()  # abstract type - will specialize later
    for (u, t) in zip(pairs(u0), t0)
        if !(u[2] isa Var)
            varname = u[1] isa Symbol ? u[1] : Symbol(name, :_u, u[1])
            if !((u[2] isa Number) || (u[2] isa Vector{<: Number}))
                throw(ArgumentError("Expected a number, a vector of numbers, or an existing continuation variable but got a $(typeof(u[2]))"))
            end
            push!(deps, Var(varname, length(u[2]), u0=u[2], t0=t))
        else
            push!(deps, u[2])
        end
    end
    _f! = f! isa EmbeddedFunction ? f! : ZeroProblem(f!)
    return ComputedFunction(_f!, (deps..., ); name=name, kwargs...)
end

zeroproblem(f!, u0; t0=nothing, kwargs...) = zeroproblem(f!, (u0,); t0=(t0,), kwargs...)

#--- AbstractMonitorFunction and MonitorFunction

abstract type AbstractMonitorFunction <: EmbeddedFunction end

mutable struct MonitorFunction{T, F} <: AbstractMonitorFunction
    f::F
    u::Var{T}
end

function monitorfunction(f, u0::NTuple{N, Var{T}}; name, active=false, initialvalue=nothing) where {N, T}
    iv = initialvalue === nothing ? f((initialvar(u).u for u in u0)...) : initialvalue
    udim = active ? 1 : 0
    u = Var(name, udim, u0=T[iv], t0=T[0])
    mfunc = MonitorFunction(f, u)
    zp = ComputedFunction(mfunc, (u, u0...), name=name, fdim=1)
end

monitorfunction(f, u0::Var; kwargs...) = monitorfunction(f, (u0,); kwargs...)

passdata(::Type{<: MonitorFunction}) = true
passproblem(::Type{<: MonitorFunction}) = true

function initialdata(zp::ComputedFunction{T, <: MonitorFunction}) where T
    mfunc = getfunc(zp)
    μ = initialvar(mfunc.u).u[1]
    return Ref(μ)
end

function evaluate!(res, mfunc::MonitorFunction, prob, data, um, u...)
    mu = isempty(um) ? data[] : um[1]
    res[1] = mfunc.f(u...) - mu
    return res
end

#--- ParameterFunction - a specialized MonitorFunction for adding continuation parameters

_identitylift(x) = x[1]

function addparameter(u::Var; name, active=false)
    return monitorfunction(_identitylift, (u,), name=name, active=active)
end

addparameter!(prob::AbstractContinuationProblem, u::Var; kwargs...) = addfunc!(getzeroproblem(prob), addparameter(u; kwargs...))

#--- FunctionCollection - functions and their dependencies

struct FunctionCollection{T, D, U, F}
    u::U
    f::F
    fdeps::Vector{Tuple{Vararg{Int64, N} where N}}
    fdim::Base.RefValue{Int64}
    fsym::Dict{Symbol, ComputedFunction{T}}
end

function FunctionCollection(T=Float64)
    FunctionCollection{T, Nothing, Vector{Var{T}}, Vector{ComputedFunction{T}}}(
        Vector{Var{T}}(),                               # u
        Vector{ComputedFunction{T}}(),                  # f
        Vector{Tuple{Vararg{Int64, N} where N}}(),      # fdeps
        Ref(zero(Int64)),                               # fdim
        Dict{Symbol, ComputedFunction{T}}(),            # fsym
    )
end

function specialize(fc::FunctionCollection{T}) where T
    u = (fc.u...,)
    f = ((specialize(f) for f in fc.f)...,)
    fdeps = fc.fdeps
    return FunctionCollection{T, (fdeps...,), typeof(u), typeof(f)}(u, f, fdeps, fc.fdim, fc.fsym)
end

function Base.show(io::IO, fc::FunctionCollection{T}) where T
    print(io, "FunctionCollection")
    (T !== Float64) && print(io, "{$T}")
    print(io, "([")
    print(io, join(["$f" for f in fc.f], ", "))
    print(io, "])")
end

fdim(fc::FunctionCollection) = fc.fdim[]

function update_fidxrange!(fc::FunctionCollection)
    last = 0
    for f in fc.f
        n = fdim(f)
        f.idxrange = (last + 1):(last + n)
        last += n
    end
    fc.fdim[] = last
    return fc
end

function addfunc!(fc::FunctionCollection{T, Nothing}, f::ComputedFunction{T}) where T
    if f in fc.f
        throw(ArgumentError("Function is already part of the collection"))
    end
    depidx = Vector{Int64}()
    for u in dependencies(f)
        ui = findfirst(==(u), fc.u)
        if ui === nothing
            ui = lastindex(push!(fc.u, u))
        end
        push!(depidx, ui)
    end
    last = fc.fdim[]
    dim = fdim(f)
    f.idx = lastindex(push!(fc.f, f))
    f.idxrange = (last + 1):(last + dim)
    fc.fdim[] = last + dim
    push!(fc.fdeps, (depidx...,))
    name = nameof(f)
    if name !== Symbol("")
        if name in keys(fc.fsym)
            @warn "Duplicate function name in collection" f
        end
        fc.fsym[name] = f
    end
    return f
end

function evaluate!(res, fc::FunctionCollection{T, Nothing}, u, prob=nothing, data=nothing) where T
    uv = [uview(u, udep.idxrange) for udep in fc.u]
    for i in eachindex(fc.f)
        args = Any[uview(res, fc.f[i].idxrange), fc.f[i]]
        if passproblem(typeof(fc.f[i]))
            push!(args, prob)
        end
        if passdata(typeof(fc.f[i]))
            push!(args, data[i])
        end
        for dep in fc.fdeps[i]
            push!(args, uv[dep])
        end
        evaluate!(args...)
    end
    return res
end

@generated function evaluate!(res, fc::FunctionCollection{T, D, U, F}, u, prob=nothing, data=nothing) where {T, D, U <: Tuple, F <: Tuple}
    body = quote
        # Construct views into u for each variable
        uv = ($((:(uview(u, fc.u[$i].idxrange)) for i in eachindex(U.parameters))...),)
    end
    # Call each of the problems
    for i in eachindex(D)
        expr = :(evaluate!(uview(res, fc.f[$i].idxrange), fc.f[$i]))
        if passproblem(F.parameters[i])
            push!(expr.args, :prob)
        end
        if passdata(F.parameters[i])
            push!(expr.args, :(data[$i]))
        end
        for j in eachindex(D[i])
            push!(expr.args, :(uv[$(D[i][j])]))
        end
        push!(body.args, expr)
    end
    # Return res
    push!(body.args, :res)
    # @show body
    body
end

function jacobian!(J, fc::FunctionCollection{T}, u, args...) where T
    # A simple forward difference
    ϵ = T(1e-6)
    @assert size(J, 1) == size(J, 2) == length(u)
    res = zeros(T, length(u))
    evaluate!(res, fc, u, args...)
    for i in eachindex(u)
        uu = u[i]
        u[i] += ϵ
        evaluate!(uview(J, :, i), fc, u, args...)
        for j in eachindex(u)
            J[j, i] = (J[j, i] - res[j])/ϵ
        end
        u[i] = uu
    end
    return J
end

function jacobian_ad(fc::FunctionCollection, u, args...) 
    ForwardDiff.jacobian((res, u)->evaluate!(res, fc, u, args...), zeros(size(u)), u)
end

initialdata(fc::FunctionCollection) = ((initialdata(f) for f in fc.f)...,)

#--- ExtendedZeroProblem - the full problem structure

struct ExtendedZeroProblem{T, E <: FunctionCollection, N <: FunctionCollection}
    u::Vector{Var{T}}
    udim::Base.RefValue{Int64}
    usym::Dict{Symbol, Var{T}}
    fsym::Dict{Symbol, ComputedFunction{T}}
    embed::E
    nonembed::N
end

function ExtendedZeroProblem(T=Float64) 
    prob = ExtendedZeroProblem(
        Vector{Var{T}}(),                               # u
        Ref(zero(Int64)),                               # udim
        Dict{Symbol, Var{T}}(),                         # usym
        Dict{Symbol, ComputedFunction{T}}(),            # fsym
        FunctionCollection(T),                          # embed
        FunctionCollection(T),                          # nonembed
    )
    addvar!(prob, Var(:allvars, 0, T=T))
    return prob
end

function specialize(zp::ExtendedZeroProblem)
    return ExtendedZeroProblem(zp.u, zp.udim, zp.usym, zp.fsym, specialize(zp.embed), specialize(zp.nonembed))
end

function Base.show(io::IO, zp::ExtendedZeroProblem{T}) where T
    print(io, "ExtendedZeroProblem")
    (T !== Float64) && print(io, "{$T}")
    print(io, "([")
    print(io, join(["$u" for u in zp.u], ", "))
    print(io, "], $(zp.embed), $(zp.nonembed))")
end

getvar(zp::ExtendedZeroProblem, u::Symbol) = zp.usym[u]
getvar(zp::ExtendedZeroProblem, u::Var) = u
getvar(prob::AbstractContinuationProblem, u) = getvar(getzeroproblem(prob), u)

getvars(zp::ExtendedZeroProblem) = collect(keys(zp.usym))
getvars(prob::AbstractContinuationProblem) = getvars(getzeroproblem(prob))

getfunc(zp::ExtendedZeroProblem, f::Symbol) = zp.fsym[f]
getfunc(zp::ExtendedZeroProblem, f::ComputedFunction) = f
getfunc(prob::AbstractContinuationProblem, f) = getfunc(getzeroproblem(prob), f)

getfuncs(zp::ExtendedZeroProblem) = collect(keys(zp.fsym))
getfuncs(prob::AbstractContinuationProblem) = getfuncs(getzeroproblem(prob))

hasvar(zp::ExtendedZeroProblem, u::Symbol) = u in keys(zp.usym)
hasvar(zp::ExtendedZeroProblem, u::Var) = u in zp.u
hasvar(prob::AbstractContinuationProblem, u) = hasvar(getzeroproblem(prob), u)

hasfunc(zp::ExtendedZeroProblem, f::Symbol) = f in keys(zp.fsym)
hasfunc(zp::ExtendedZeroProblem, f::ComputedFunction) = f in zp.f
hasfunc(prob::AbstractContinuationProblem, f) = hasfunc(getzeroproblem(prob), f)

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
fdim(zp::ExtendedZeroProblem) = fdim(zp, EmbeddedFunction)
fdim(zp::ExtendedZeroProblem, ::Type{EmbeddedFunction}) = fdim(zp.embed)
fdim(zp::ExtendedZeroProblem, ::Type{NonEmbeddedFunction}) = fdim(zp.nonembed)
fdim(prob::AbstractContinuationProblem, args...) = fdim(getzeroproblem(prob), args...)

function update_uidxrange!(u::Var, last::Int64)
    n = udim(u)
    if u.parent === nothing
        ui = (last + 1):(last + n)
        last += n
    else
        parentidx = uidxrange(u.parent)
        start = (u.offset < 0) ? (parentidx[end] + u.offset + 1) : (parentidx[1] + u.offset)
        ui = start:(start + n - 1)
    end
    u.idxrange = ui
    return last
end

function update_uidxrange!(zp::ExtendedZeroProblem)
    last = 0
    for u in zp.u
        last = update_uidxrange!(u, last)
    end
    zp.udim[] = last
    zp.u[1].idxrange = 1:last
    return zp
end

function addvar!(zp::ExtendedZeroProblem, u::Var)
    if !(u in zp.u)
        up = u.parent
        if (up !== nothing) && !(up in zp.u)
            throw(ArgumentError("Parent variable is not contained in the zero problem"))
        end
        u.idx = lastindex(push!(zp.u, u))
        last = update_uidxrange!(u, zp.udim[])
        zp.udim[] = last
        zp.u[1].idxrange = 1:last
        name = nameof(u)
        if (name !== Symbol(""))
            if name in keys(zp.usym)
                @warn "Duplicate variable name in ExtendedZeroProblem" u
            end
            zp.usym[name] = u
        end
    end
    return u
end

addvar!(prob::AbstractContinuationProblem, u::Var) = addvar!(getzeroproblem(prob), u)

function initialvar(zp::ExtendedZeroProblem{T}) where T
    ndim = udim(zp)
    u = zeros(T, ndim)
    t = zeros(T, ndim)
    for udep in Iterators.drop(zp.u, 1)  # First var is :allvars
        if parent(udep) === nothing
            u[udep.idxrange] .= udep.u0
            t[udep.idxrange] .= udep.t0
        end
    end
    return (u=u, TS=t)
end

function setvaractive!(zp::ExtendedZeroProblem, u::Var, active::Bool)
    u.len = active ? 1 : 0
    update_uidxrange!(zp)
    return
end

setvaractive!(zp::ExtendedZeroProblem, u::Symbol, active::Bool) = setvaractive!(zp, getvar(zp, u), active)
setvaractive!(prob::AbstractContinuationProblem, u, active) = setvaractive!(getzeroproblem(prob), u, active)

isvaractive(u) = u.len > 0

function addfunc!(zp::ExtendedZeroProblem, f::ComputedFunction{<:Any, F}) where F
    if nameof(f) in keys(zp.fsym)
        @warn "Duplicate function name in ExtendedZeroProblem" f
    end
    for u in dependencies(f)
        addvar!(zp, u)
    end
    zp.fsym[nameof(f)] = f
    if F <: EmbeddedFunction
        addfunc!(zp.embed, f)
    elseif F <: NonEmbeddedFunction
        addfunc!(zp.nonembed, f)
    else
        throw(ArgumentError("Unknown function type - should be a subtype of either an EmbeddedFunction or a NonEmbeddedFunction"))
    end
end

addfunc!(prob::AbstractContinuationProblem, f::ComputedFunction) = addfunc!(getzeroproblem(prob), f)

evaluate_embedded!(res, zp::ExtendedZeroProblem, u, args...) = evaluate!(res, zp.embed, u, args...)
evaluate_nonembedded!(res, zp::ExtendedZeroProblem, u, args...) = evaluate!(res, zp.nonembed, u, args...)

jacobian_ad(zp::ExtendedZeroProblem, u, args...) = jacobian_ad(zp.embed, u, args...)

initialdata_embedded(zp::ExtendedZeroProblem) = initialdata(zp.embed)
initialdata_nonembedded(zp::ExtendedZeroProblem) = initialdata(zp.nonembed)

#--- NonEmbedded functions

abstract type AbstractRegularFunction <: NonEmbeddedFunction end

struct RegularFunction{T, F} <: AbstractRegularFunction
    f::F
end

(rf::RegularFunction)(u...) = rf.f(u...)

function regularfunction(f, u::NTuple{N, Var{T}}; name, kwargs...) where {N, T}
    ComputedFunction(RegularFunction{T}(f), u; name=name, kwargs...)
end

abstract type AbstractSingularFunction <: NonEmbeddedFunction end

struct SingularFunction{T, F} <: AbstractSingularFunction
    f::F
end

(sf::SingularFunction)(u...) = sf.f(u...)

function singularfunction(f, u::NTuple{N, Var{T}}; name, kwargs...) where {N, T}
    ComputedFunction(SingularFunction{T}(f), u; name=name, kwargs...)
end

end # module
