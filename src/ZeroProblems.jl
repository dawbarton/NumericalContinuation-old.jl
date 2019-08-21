module ZeroProblems

using UnsafeArrays: uview
using ..NumericalContinuation: AbstractContinuationProblem, getzeroproblem
import ..NumericalContinuation: specialize, numtype

import ForwardDiff

#--- Exports

export ExtendedZeroProblem, ZeroProblem, ZeroProblem!, Var, MonitorFunction
export residual!, fdim, udim, fidx, uidx, fidxrange, uidxrange, dependencies, 
    addparameter, addparameter!, getvar, getproblem, hasvar, hasproblem,
    setvaractive!, isvaractive

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

#--- Common helpers

function constructdeps(u0, t0, T; name)
    # Construct continuation variables as necessary
    deps = Vector{Var}()  # abstract type - will specialize when returning
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
    _T = (T === nothing) ? numtype(first(deps)) : T
    vars = Dict{Symbol, Var{_T}}()
    for dep in deps
        if nameof(dep) in keys(vars)
            @warn "Duplicate variable name" dep
        end
        vars[nameof(dep)] = dep
    end
    return (convert(Vector{Var{_T}}, deps), vars)
end

#--- ZeroProblem

"""
    ZeroProblem{T, F}
"""
mutable struct ZeroProblem{T, F}
    name::Symbol
    deps::Vector{Var{T}}
    f!::F
    fdim::Int64
    vars::Dict{Symbol, Var{T}}
    idx::Int64
    idxrange::UnitRange{Int64}
end

function ZeroProblem(f, u0::Union{Tuple, NamedTuple}; T=nothing, fdim=0, t0=Iterators.repeated(nothing), name=:zero, inplace=false)
    deps, vars = constructdeps(u0, t0, T, name=name)
    # Determine whether f is in-place or not
    if inplace
        f! = f
        if fdim == 0
            throw(ArgumentError("For in-place functions the number of dimensions (fdim) must be specified"))
        end
    else
        f! = (res, u...) -> res .= f(u...)
        if fdim == 0
            res = f((initialdata(u).u for u in deps)...)
            fdim = length(res)
        end
    end
    # Construct the continuation variables
    return ZeroProblem(name, deps, f!, fdim, vars, 0, 0:0)
end

ZeroProblem(f, u0; t0=nothing, kwargs...) = ZeroProblem(f, (u0,); t0=(t0,), kwargs...)

function ZeroProblem!(prob::AbstractContinuationProblem, args...; name=:zero, kwargs...)
    subprob = ZeroProblem(args...; name=nextproblemname(prob, name), kwargs...)
    push!(prob, subprob)
    return subprob
end

function Base.show(io::IO, @nospecialize prob::ZeroProblem{T, F}) where {T, F}
    _T = T === Float64 ? "" : "$(nameof(T)), "
    name = nameof(prob) === Symbol("") ? "UNNAMED" : ":$(nameof(prob))"
    print(io, "ZeroProblem{$_T$(nameof(F))}($name, $(fdim(prob)))")
end

Base.nameof(prob::ZeroProblem) = prob.name
dependencies(prob::ZeroProblem) = prob.deps
fdim(prob::ZeroProblem) = prob.fdim
initialdata(prob::ZeroProblem) = initialdata(prob.f!)
numtype(prob::ZeroProblem{T}) where T = T
Base.getindex(prob::ZeroProblem, idx::Integer) = getindex(prob.deps, idx)
Base.getindex(prob::ZeroProblem, sym::Symbol) = prob.vars[sym]
fidx(prob::ZeroProblem) = prob.idx
fidxrange(prob::ZeroProblem) = prob.idxrange
getfunc(prob::ZeroProblem) = prob.f!

residual!(res, f!, u...) = f!(res, u...)
residual!(res, prob::ZeroProblem, u...) = residual!(res, prob.f!, u...)

passdata(::Type{<: ZeroProblem{T, F}}) where {T, F} = passdata(F)
passproblem(::Type{<: ZeroProblem{T, F}}) where {T, F} = passproblem(F)

#--- MonitorFunction

mutable struct MonitorFunction{T, F}
    f::F
    u::Var{T}
end

function monitorfunction(f, u0::NTuple{N, Var{T}}; name=:mfunc, active=false) where {N, T}
    udim = active ? 1 : 0
    u = Var(name, udim, T=T)
    mfunc = MonitorFunction(f, u)
    zp = ZeroProblem(mfunc, (u, u0...), name=name, fdim=1, inplace=true)
end
monitorfunction(f, u0; kwargs...) = monitorfunction(f, (u0,); kwargs...)

function monitorfunction!(prob::AbstractContinuationProblem, args...; name=:mfunc, kwargs...)
    subprob = monitorfunction(args...; name=nextproblemname(prob, name), kwargs...)
    push!(prob, subprob)
    return subprob
end

passdata(::Type{<: MonitorFunction}) = true
passproblem(::Type{<: MonitorFunction}) = true

function initialdata(zp::ZeroProblem{T, <: MonitorFunction}) where T
    mfunc = getfunc(zp)
    μ = T(mfunc.f((initialdata(u).u for u in Iterators.drop(dependencies(zp), 1))...))
    fdata = initialdata(mfunc.f)
    return (Ref(μ), fdata)
end

function residual!(res, mfunc::MonitorFunction, prob, data, um, u...)
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

addparameter!(prob::AbstractContinuationProblem, u::Var; kwargs...) = push!(getzeroproblem(prob), addparameter(u; kwargs...))

#--- ExtendedZeroProblem - the full problem structure

struct ExtendedZeroProblem{T, D, U, Φ}
    u::U
    udim::Base.RefValue{Int64}
    usym::Dict{Symbol, Var{T}}
    ϕ::Φ
    ϕdeps::Vector{Tuple{Vararg{Int64, N} where N}}
    ϕdim::Base.RefValue{Int64}
    ϕsym::Dict{Symbol, ZeroProblem{T}}
end

function ExtendedZeroProblem(T=Float64) 
    prob = ExtendedZeroProblem{T, Nothing, Vector{Var{T}}, Vector{ZeroProblem{T}}}(
        Vector{Var{T}}(),                               # u
        Ref(zero(Int64)),                               # udim
        Dict{Symbol, Var{T}}(),                         # usym
        Vector{ZeroProblem{T}}(),                       # ϕ
        Vector{Tuple{Vararg{Int64, N} where N}}(),      # ϕdeps
        Ref(zero(Int64)),                               # ϕdim
        Dict{Symbol, ZeroProblem{T}}(),                 # ϕsym
    )
    push!(prob, Var(:allvars, 0, T=T))
    return prob
end

function ExtendedZeroProblem(probs::Vector{<: ZeroProblem{T}}) where T
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

getvar(zp::ExtendedZeroProblem, u::Symbol) = zp.usym[u]
getvar(zp::ExtendedZeroProblem, u::Var) = u
getvar(prob::AbstractContinuationProblem, u) = getvar(getzeroproblem(prob), u)
getproblem(zp::ExtendedZeroProblem, f::Symbol) = zp.ϕsym[f]
getproblem(zp::ExtendedZeroProblem, f::ZeroProblem) = f
getproblem(prob::AbstractContinuationProblem, f) = getproblem(getzeroproblem(prob), f)

hasvar(zp::ExtendedZeroProblem, u::Symbol) = u in keys(zp.usym)
hasvar(zp::ExtendedZeroProblem, u::Var) = u in zp.u
hasvar(prob::AbstractContinuationProblem, u) = hasvar(getzeroproblem(prob), u)
hasproblem(zp::ExtendedZeroProblem, f::Symbol) = f in keys(zp.ϕsym)
hasproblem(zp::ExtendedZeroProblem, f::ZeroProblem) = f in zp.ϕ
hasproblem(prob::AbstractContinuationProblem, f) = hasproblem(getzeroproblem(prob), f)

function nextproblemname(zp::ExtendedZeroProblem, f::Symbol)
    if !hasproblem(zp, f)
        return f
    else
        i = 2
        while true
            next = Symbol(f, i)
            if !hasproblem(zp, next)
                return next
            end
            i += 1
        end
    end
end

nextproblemname(prob::AbstractContinuationProblem, f) = nextproblemname(getzeroproblem(prob), f)

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
u[uidxrange(prob, ui)]  # as frequently as necessary (fast)
```
"""
uidx(zp::ExtendedZeroProblem, u) = uidx(getvar(zp, u))
uidx(prob::AbstractContinuationProblem, x) = uidx(getzeroproblem(prob), x)

"""
    uidxrange(prob, i::Integer)

Return the index of the continuation variable within the solution vector. (May
change during continuation, for example if adaptive meshing is used.)
"""
uidxrange(zp::ExtendedZeroProblem, i::Integer) = uidxrange(zp.u[i])

"""
    fidx(prob, prob::ZeroProblem)

Return the index of the sub-problem within the internal structures. This will
not change during continuation and so can be stored for fast indexing
throughout the continuation run.

# Example

```
fi = fidx(prob, myproblem)  # once at the start of the continuation run (slow)
res[fidxrange(prob, fi)]  # as frequently as necessary (fast)
```
"""
fidx(zp::ExtendedZeroProblem, prob) = fidx(getproblem(zp, prob))
fidx(prob::AbstractContinuationProblem, x) = fidx(getzeroproblem(prob), x)

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

function Base.push!(zp::ExtendedZeroProblem{T, Nothing}, u::Var{T}) where T
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
    return zp
end

Base.push!(prob::AbstractContinuationProblem{T}, u::Var{T}) where T = push!(getzeroproblem(prob), u)

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

function Base.push!(zp::ExtendedZeroProblem{T, Nothing}, prob::ZeroProblem{T}) where T
    if prob in zp.ϕ
        throw(ArgumentError("Problem is already part of the zero problem"))
    end
    depidx = Vector{Int64}()
    for u in dependencies(prob)
        push!(zp, u)
        push!(depidx, uidx(u))
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

Base.push!(prob::AbstractContinuationProblem{T}, zp::ZeroProblem{T}) where T = push!(getzeroproblem(prob), zp)

function residual!(res, zp::ExtendedZeroProblem{T, Nothing}, u, prob=nothing, data=nothing) where T
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
        residual!(args...)
    end
    return res
end

@generated function residual!(res, zp::ExtendedZeroProblem{T, D, U, Φ}, u, prob=nothing, data=nothing) where {T, D, U <: Tuple, Φ <: Tuple}
    body = quote
        # Construct views into u for each variable
        uv = ($((:(uview(u, zp.u[$i].idxrange)) for i in eachindex(U.parameters))...),)
    end
    # Call each of the problems
    for i in eachindex(D)
        expr = :(residual!(uview(res, zp.ϕ[$i].idxrange), zp.ϕ[$i]))
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

end # module
