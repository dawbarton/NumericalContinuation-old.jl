module ZeroProblems

using UnsafeArrays: uview
using ..NumericalContinuation: AbstractContinuationProblem, getzeroproblem
import ..NumericalContinuation: specialize, numtype

import ForwardDiff

#--- Exports

export ExtendedZeroProblem, ComputedFunction, ComputedFunction!, Var, MonitorFunction,
    EmbeddedFunction, NonEmbeddedFunction
export evaluate!, fdim, udim, fidxrange, uidxrange, dependencies, addparameter, 
    addparameter!, getvar, getfunc, hasvar, hasfunc, setvaractive!, 
    isvaractive, zeroproblem, zeroproblem!, monitorfunction, monitorfunction!,
    addfunc!, addvar!

#--- Forward definitions

"""
    dependencies(z)

Return the variable dependencies of a zero problem.
"""
function dependencies end

"""
    passproblem(z)

A trait to determine whether the full problem structure is passed down to a
particular subtype of AbstractZeroProblem. The default is false.

Also see `passdata`.

# Example

A ComputedFunction containing the pseudo-arclength equation might require the
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

A ComputedFunction containing collocation equations might require the time
discretization which it stores in its own data structure and so it defines

```
passdata(z::Type{Collocation}) = true
```
"""
passdata(z) = false

"""
    evaluate!(res, [J], z, u, [prob])

Return the residual (inplace), and optionally the Jacobian, of the ExtendedZeroProblem
z with the input u. Some ExtendedZeroProblems also require the problem structure
`prob` to be passed.
"""
function evaluate! end

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
    initialdata(prob)

Return the initial data (solution, tangent, toolbox data) used for initialising
the continuation.
"""
initialdata(prob) = nothing

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
initialdata(u::Var) = (u=u.u0, TS=u.t0)
numtype(u::Var{T}) where T = T
Base.parent(u::Var) = u.parent
uidx(u::Var) = u.idx
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
initialdata(prob::ComputedFunction) = initialdata(prob.f!)
numtype(prob::ComputedFunction{T}) where T = T
Base.getindex(prob::ComputedFunction, idx::Integer) = getindex(prob.deps, idx)
Base.getindex(prob::ComputedFunction, sym::Symbol) = prob.vars[sym]
fidx(prob::ComputedFunction) = prob.idx
fidxrange(prob::ComputedFunction) = prob.idxrange
getfunc(prob::ComputedFunction) = prob.f!

evaluate!(res, f!, u...) = f!(res, u...)
evaluate!(res, prob::ComputedFunction, u...) = evaluate!(res, prob.f!, u...)

passdata(::Type{<: ComputedFunction{T, F}}) where {T, F} = passdata(F)
passproblem(::Type{<: ComputedFunction{T, F}}) where {T, F} = passproblem(F)

#--- ZeroProblem

"""
    ZeroProblem{T, F}

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

#--- MonitorFunction

mutable struct MonitorFunction{T, F} <: EmbeddedFunction
    f::F
    u::Var{T}
end

function monitorfunction(f, u0::NTuple{N, Var{T}}; name, active=false, initialvalue=nothing) where {N, T}
    iv = initialvalue === nothing ? f((initialdata(u).u for u in u0)...) : initialvalue
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
    μ = initialdata(mfunc.u).u[1]
    fdata = initialdata(mfunc.f)
    return (Ref(μ), fdata)
end

function evaluate!(res, mfunc::MonitorFunction, prob, data, um, u...)
    μ = isempty(um) ? data[1][] : um[1]
    _passdata = passdata(typeof(mfunc.f))
    _passprob = passproblem(typeof(mfunc.f))
    if _passdata
        if _passprob
            res[1] = mfunc.f(prob, data[2], u...) - μ
        else
            res[1] = mfunc.f(data[2], u...) - μ
        end
    elseif _passprob
        res[1] = mfunc.f(prob, u...) - μ
    else
        res[1] = mfunc.f(u...) - μ
    end
    return nothing
end

#--- ParameterFunction - a specialized MonitorFunction for adding continuation parameters

_identitylift(x) = x[1]

function addparameter(u::Var; name, active=false)
    return monitorfunction(_identitylift, (u,), name=name, active=active)
end

addparameter!(prob::AbstractContinuationProblem, u::Var; kwargs...) = addfunc!(getzeroproblem(prob), addparameter(u; kwargs...))

#--- ExtendedZeroProblem - the full problem structure

struct ExtendedZeroProblem{T, D, U, Φ}
    u::U
    udim::Base.RefValue{Int64}
    usym::Dict{Symbol, Var{T}}
    ϕ::Φ
    ϕdeps::Vector{Tuple{Vararg{Int64, N} where N}}
    ϕdim::Base.RefValue{Int64}
    ϕsym::Dict{Symbol, ComputedFunction{T}}
end

function ExtendedZeroProblem(T=Float64) 
    prob = ExtendedZeroProblem{T, Nothing, Vector{Var{T}}, Vector{ComputedFunction{T}}}(
        Vector{Var{T}}(),                               # u
        Ref(zero(Int64)),                               # udim
        Dict{Symbol, Var{T}}(),                         # usym
        Vector{ComputedFunction{T}}(),                  # ϕ
        Vector{Tuple{Vararg{Int64, N} where N}}(),      # ϕdeps
        Ref(zero(Int64)),                               # ϕdim
        Dict{Symbol, ComputedFunction{T}}(),            # ϕsym
    )
    addvar!(prob, Var(:allvars, 0, T=T))
    return prob
end

function ExtendedZeroProblem(probs::Vector{<: ComputedFunction{T}}) where T
    zp = ExtendedZeroProblem(T)
    for prob in probs
        push!(zp, prob)
    end
    return zp
end

function specialize(zp::ExtendedZeroProblem{T}) where T
    u = (zp.u...,)
    ϕ = ((specialize(ϕ) for ϕ in zp.ϕ)...,)
    ϕdeps = zp.ϕdeps
    return ExtendedZeroProblem{T, (ϕdeps...,), typeof(u), typeof(ϕ)}(u, zp.udim, zp.usym, ϕ, ϕdeps, zp.ϕdim, zp.ϕsym)
end

function Base.show(io::IO, zp::ExtendedZeroProblem{T}) where T
    print(io, "ExtendedZeroProblem")
    (T !== Float64) && print(io, "{$T}")
    print(io, "(Var[")
    print(io, join(["$u" for u in zp.u], ", "))
    print(io, "], ComputedFunction[")
    print(io, join(["$ϕ" for ϕ in zp.ϕ], ", "))
    print(io, "])")
end

getvar(zp::ExtendedZeroProblem, u::Symbol) = zp.usym[u]
getvar(zp::ExtendedZeroProblem, u::Var) = u
getvar(prob::AbstractContinuationProblem, u) = getvar(getzeroproblem(prob), u)
getfunc(zp::ExtendedZeroProblem, f::Symbol) = zp.ϕsym[f]
getfunc(zp::ExtendedZeroProblem, f::ComputedFunction) = f
getfunc(prob::AbstractContinuationProblem, f) = getfunc(getzeroproblem(prob), f)

hasvar(zp::ExtendedZeroProblem, u::Symbol) = u in keys(zp.usym)
hasvar(zp::ExtendedZeroProblem, u::Var) = u in zp.u
hasvar(prob::AbstractContinuationProblem, u) = hasvar(getzeroproblem(prob), u)
hasfunc(zp::ExtendedZeroProblem, f::Symbol) = f in keys(zp.ϕsym)
hasfunc(zp::ExtendedZeroProblem, f::ComputedFunction) = f in zp.ϕ
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
fdim(zp::ExtendedZeroProblem) = zp.ϕdim[]
fdim(prob::AbstractContinuationProblem) = fdim(getzeroproblem(prob))

"""
    uidxrange(prob, i::Integer)

Return the index of the continuation variable within the solution vector. (May
change during continuation, for example if adaptive meshing is used.)
"""
uidxrange(zp::ExtendedZeroProblem, i::Integer) = uidxrange(zp.u[i])

"""
    fidxrange(prob, i::Integer)

Return the index of the sub-problem within the residual vector. (May change
during continuation, for example if adaptive meshing is used.)
"""
fidxrange(zp::ExtendedZeroProblem, i::Integer) = fidxrange(zp.ϕ[i])

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

function addvar!(zp::ExtendedZeroProblem{T, Nothing}, u::Var{T}) where T
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

addvar!(prob::AbstractContinuationProblem{T}, u::Var{T}) where T = addvar!(getzeroproblem(prob), u)

function update_ϕi!(zp::ExtendedZeroProblem)
    last = 0
    for ϕ in zp.ϕ
        n = fdim(ϕ)
        ϕ.idxrange = (last + 1):(last + n)
        last += n
    end
    zp.ϕdim[] = last
    return zp
end

function addfunc!(zp::ExtendedZeroProblem{T, Nothing}, prob::ComputedFunction{T}) where T
    if prob in zp.ϕ
        throw(ArgumentError("Problem is already part of the zero problem"))
    end
    depidx = Vector{Int64}()
    for u in dependencies(prob)
        push!(depidx, uidx(addvar!(zp, u)))
    end
    last = zp.ϕdim[]
    ϕdim = fdim(prob)
    prob.idx = lastindex(push!(zp.ϕ, prob))
    prob.idxrange = (last + 1):(last + ϕdim)
    zp.ϕdim[] = last + ϕdim
    push!(zp.ϕdeps, (depidx...,))
    name = nameof(prob)
    if name !== Symbol("")
        if name in keys(zp.ϕsym)
            @warn "Duplicate problem name in ExtendedZeroProblem" prob
        end
        zp.ϕsym[name] = prob
    end
    return zp
end

addfunc!(prob::AbstractContinuationProblem{T}, zp::ComputedFunction{T}) where T = addfunc!(getzeroproblem(prob), zp)

function evaluate!(res, zp::ExtendedZeroProblem{T, Nothing}, u, prob=nothing, data=nothing) where T
    uv = [uview(u, udep.idxrange) for udep in zp.u]
    for i in eachindex(zp.ϕ)
        args = Any[uview(res, zp.ϕ[i].idxrange), zp.ϕ[i]]
        if passproblem(typeof(zp.ϕ[i]))
            push!(args, prob)
        end
        if passdata(typeof(zp.ϕ[i]))
            push!(args, data[i])
        end
        for dep in zp.ϕdeps[i]
            push!(args, uv[dep])
        end
        evaluate!(args...)
    end
    return res
end

@generated function evaluate!(res, zp::ExtendedZeroProblem{T, D, U, Φ}, u, prob=nothing, data=nothing) where {T, D, U <: Tuple, Φ <: Tuple}
    body = quote
        # Construct views into u for each variable
        uv = ($((:(uview(u, zp.u[$i].idxrange)) for i in eachindex(U.parameters))...),)
    end
    # Call each of the problems
    for i in eachindex(D)
        expr = :(evaluate!(uview(res, zp.ϕ[$i].idxrange), zp.ϕ[$i]))
        if passproblem(Φ.parameters[i])
            push!(expr.args, :prob)
        end
        if passdata(Φ.parameters[i])
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

evaluate!(res, prob::AbstractContinuationProblem, u, args...) = evaluate!(res, getzeroproblem(prob), u, args...)

function jacobian!(J, zp::ExtendedZeroProblem{T}, u, args...) where T
    # A simple forward difference
    ϵ = T(1e-6)
    @assert size(J, 1) == size(J, 2) == length(u)
    res = zeros(T, length(u))
    evaluate!(res, zp, u, args...)
    for i in eachindex(u)
        uu = u[i]
        u[i] += ϵ
        evaluate!(uview(J, :, i), zp, u, args...)
        for j in eachindex(u)
            J[j, i] = (J[j, i] - res[j])/ϵ
        end
        u[i] = uu
    end
    return J
end

function jacobian_ad(zp::ExtendedZeroProblem, u, args...) 
    ForwardDiff.jacobian((res, u)->evaluate!(res, zp, u, args...), zeros(size(u)), u)
end

function initialdata(zp::ExtendedZeroProblem{T}) where T
    ndim = udim(zp)
    u = zeros(T, ndim)
    t = zeros(T, ndim)
    for udep in Iterators.drop(zp.u, 1)  # First var is :allvars
        if parent(udep) === nothing
            u[udep.idxrange] .= udep.u0
            t[udep.idxrange] .= udep.t0
        end
    end
    data = ((initialdata(ϕ) for ϕ in zp.ϕ)...,)
    return (u=u, TS=t, data=data)
end

function setvaractive!(zp::ExtendedZeroProblem, u::Var, active::Bool)
    u.len = active ? 1 : 0
    update_uidxrange!(zp)
    return
end
setvaractive!(zp::ExtendedZeroProblem, u::Symbol, active::Bool) = setvaractive!(zp, getvar(zp, u), active)
setvaractive!(prob::AbstractContinuationProblem, u, active) = setvaractive!(getzeroproblem(prob), u, active)

isvaractive(u) = u.len > 0

#--- NonEmbedded functions

struct RegularFunction{T, F} <: NonEmbeddedFunction
    f::F
end

(rf::RegularFunction)(u...) = rf.f(u...)

function regularfunction(f, u::NTuple{N, Var{T}}; name=:reg, kwargs...) where {N, T}
    ComputedFunction(RegularFunction{T}(f), u; name=name, kwargs...)
end

struct SingularFunction{T, F} <: NonEmbeddedFunction
    f::F
end

(sf::SingularFunction)(u...) = sf.f(u...)

function singularfunction(f, u::NTuple{N, Var{T}}; name=:sing, kwargs...) where {N, T}
    ComputedFunction(SingularFunction{T}(f), u; name=name, kwargs...)
end

end # module
