module ZeroProblems

using UnsafeArrays: uview
using ..NumericalContinuation: AbstractContinuationProblem, getzeroproblem
import ..NumericalContinuation: specialize, numtype

import ForwardDiff

#--- Exports

export ExtendedZeroProblem, ZeroProblem, ZeroProblem!, Var, MonitorFunction, 
    ParameterFunction
export residual!, fdim, udim, fidx, uidx, dependencies, addparameter, 
    addparameter!, getvar

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
    initialdata(prob)

Return the initial data (solution, tangent, toolbox data) used for initialising
the continuation.
"""
function initialdata end

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
initialdata(u::Var) = (u=u.u0, TS=u.t0)
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
initialdata(prob::AbstractZeroProblem) = (data=nothing,)
numtype(prob::AbstractZeroProblem{T}) where T = T
Base.getindex(prob::AbstractZeroProblem, idx::Integer) = getindex(prob.deps, idx)
Base.getindex(prob::AbstractZeroProblem, sym::Symbol) = prob.vars[sym]

function Base.show(io::IO, @nospecialize prob::AbstractZeroProblem)
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
            res = f((initialdata(u).u for u in deps)...)
            fdim = length(res)
        end
    end
    # Construct the continuation variables
    return ZeroProblem(name, deps, f!, fdim, vars)
end

ZeroProblem(f, u0; t0=nothing, kwargs...) = ZeroProblem(f, (u0,); t0=(t0,), kwargs...)

function ZeroProblem!(prob::AbstractContinuationProblem, args...; name=:zero, kwargs...)
    subprob = ZeroProblem(args...; name=nextproblemname(prob, name), kwargs...)
    push!(prob, subprob)
    return subprob
end

residual!(res, zp::ZeroProblem, u...) = zp.f!(res, u...)

#--- MonitorFunction & AbstractMonitorFunction

abstract type AbstractMonitorFunction{T} <: AbstractZeroProblem{T} end

fdim(mfunc::AbstractMonitorFunction) = 1

mutable struct MonitorFunction{T, F} <: AbstractMonitorFunction{T}
    name::Symbol
    deps::Vector{Var{T}}
    f::F
    vars::Dict{Symbol, Var{T}}
    μ::Var{T}
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
    return MonitorFunction(name, deps, f, vars, μ, active)
end
MonitorFunction(f, u0; t0=nothing, kwargs...) = MonitorFunction(f, (u0,); t0=(t0,), kwargs...)

passdata(mfunc::MonitorFunction) = true
initialdata(mfunc::MonitorFunction) = (data=Ref(mfunc.f((initialdata(u).u for u in Iterators.drop(mfunc.deps, 1))...)),)
isvaractive(mfunc::MonitorFunction) = mfunc.active
getvar(mfunc::MonitorFunction) = mfunc.μ

function residual!(res, mfunc::MonitorFunction, data, um, u...)
    μ = isempty(um) ? data[] : um[1]
    res[1] = mfunc.f(u...) - μ
end

function setvaractive!(mfunc::MonitorFunction, active::Bool)
    mfunc.active = active
    mfunc.μ.len = active ? 1 : 0
    return
end

#--- ParameterFunction - a specialized MonitorFunction for adding continuation parameters

_identitylift(x) = x[1]

const ParameterFunction{T} = MonitorFunction{T, typeof(_identitylift)}

function addparameter(u::Var; name, active=false)
    return MonitorFunction(_identitylift, (u,), name=name, active=active)
end

addparameter!(prob::AbstractContinuationProblem, u::Var; kwargs...) = push!(getzeroproblem(prob), addparameter(u; kwargs...))

#--- VarInfo & ProblemInfo

mutable struct VarInfo{T}
    u::Var{T}
    idx::Int64
    deps::Set{AbstractZeroProblem{T}}
    mfunc::Union{Nothing, MonitorFunction{T}}
end
VarInfo(u::Var{T}, idx::Int64) where T = VarInfo(u, idx, Set{AbstractZeroProblem{T}}(), nothing)
getvar(info::VarInfo) = info.u
uidx(info::VarInfo) = info.idx
dependencies(info::VarInfo) = info.deps
getmfunc(info::VarInfo) = info.mfunc
getmfunc(prob, u::Var) = getmfunc(getvarinfo(prob, u))

function adddependency!(info::VarInfo, prob::AbstractZeroProblem)
    push!(info.deps, prob)
    if (prob isa AbstractMonitorFunction) && (getvar(prob) === info.u)
        info.mfunc = prob
    end
    return
end

struct ProblemInfo{T}
    prob::AbstractZeroProblem{T}
    idx::Int64
end
getproblem(info::ProblemInfo) = info.prob
fidx(info::ProblemInfo) = info.idx

#--- ExtendedZeroProblem - the full problem structure

struct ExtendedZeroProblem{T, D, U, Φ}
    u::U
    ui::Vector{UnitRange{Int64}}
    udim::Base.RefValue{Int64}
    uinfo::Vector{VarInfo{T}}
    usym::Dict{Symbol, VarInfo{T}}
    uvar::Dict{Var{T}, VarInfo{T}}
    ϕ::Φ
    ϕi::Vector{UnitRange{Int64}}
    ϕdeps::Vector{Tuple{Vararg{Int64, N} where N}}
    ϕdim::Base.RefValue{Int64}
    ϕsym::Dict{Symbol, ProblemInfo{T}}
    ϕprob::Dict{AbstractZeroProblem{T}, ProblemInfo{T}}
end

ExtendedZeroProblem(T=Float64) = 
    ExtendedZeroProblem{T, Nothing, Vector{Var{T}}, Vector{AbstractZeroProblem{T}}}(
        Vector{Var{T}}(),                               # u
        Vector{UnitRange{Int64}}(),                     # ui
        Ref(zero(Int64)),                               # udim
        Vector{VarInfo{T}}(),                           # uinfo
        Dict{Symbol, VarInfo{T}}(),                     # usym
        Dict{Var{T}, VarInfo{T}}(),                     # uvar
        Vector{AbstractZeroProblem{T}}(),               # ϕ
        Vector{UnitRange{Int64}}(),                     # ϕi
        Vector{Tuple{Vararg{Int64, N} where N}}(),      # ϕdeps
        Ref(zero(Int64)),                               # ϕdim
        Dict{Symbol, ProblemInfo{T}}(),                 # ϕsym
        Dict{AbstractZeroProblem{T}, ProblemInfo{T}}(), # ϕprob
    )

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
    return ExtendedZeroProblem{T, (ϕdeps...,), typeof(u), typeof(ϕ)}(u, ui, zp.udim, zp.uinfo, zp.usym, zp.uvar, ϕ, ϕi, ϕdeps, zp.ϕdim, zp.ϕsym, zp.ϕprob)
end

getvarinfo(zp::ExtendedZeroProblem, u::Var) = zp.uvar[u]
getvarinfo(zp::ExtendedZeroProblem, u::Symbol) = zp.usym[u]
getprobleminfo(zp::ExtendedZeroProblem, f::AbstractZeroProblem) = zp.ϕprob[f]
getprobleminfo(zp::ExtendedZeroProblem, f::Symbol) = zp.ϕsym[f]

hasvar(zp::ExtendedZeroProblem, u::Symbol) = u in keys(zp.usym)
hasproblem(zp::ExtendedZeroProblem, f::Symbol) = f in keys(zp.ϕsym)

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
u[uidx(prob, ui)]  # as frequently as necessary (fast)
```
"""
uidx(zp::ExtendedZeroProblem, u) = uidx(getvarinfo(zp, u))
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
fidx(zp::ExtendedZeroProblem, prob) = fidx(getprobleminfo(zp, prob))
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
        idx = lastindex(push!(zp.u, u))
        (ui, last) = update_ui(zp, u, zp.udim[])
        push!(zp.ui, ui)
        zp.udim[] = last
        uinfo = VarInfo(u, idx)
        push!(zp.uinfo, uinfo)
        if nameof(u) in keys(zp.usym)
            @warn "Duplicate variable name in ExtendedZeroProblem" u
        end
        zp.usym[nameof(u)] = uinfo
        zp.uvar[u] = uinfo
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
        uinfo = getvarinfo(zp, dep)
        push!(depidx, uidx(uinfo))
        adddependency!(uinfo, prob)
    end
    idx = lastindex(push!(zp.ϕ, prob))
    last = zp.ϕdim[]
    ϕdim = fdim(prob)
    push!(zp.ϕi, (last + 1):(last + ϕdim))
    zp.ϕdim[] = last + ϕdim
    push!(zp.ϕdeps, (depidx...,))
    probinfo = ProblemInfo(prob, idx)
    if nameof(prob) in keys(zp.ϕsym)
        @warn "Duplicate problem name in ExtendedZeroProblem" prob
    end
    zp.ϕsym[nameof(prob)] = probinfo
    zp.ϕprob[prob] = probinfo
    return zp
end

Base.push!(prob::AbstractContinuationProblem{T}, zp::AbstractZeroProblem{T}) where T = push!(getzeroproblem(prob), zp)

function residual!(res, zp::ExtendedZeroProblem{T, Nothing}, u, prob=nothing, data=nothing) where T
    uv = [uview(u, zp.ui[i]) for i in eachindex(zp.ui)]
    for i in eachindex(zp.ϕ)
        args = Any[uview(res, zp.ϕi[i]), zp.ϕ[i]]
        if passproblem(zp.ϕ[i])
            push!(args, prob)
        end
        if passdata(zp.ϕ[i])
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
        if passproblem(Φ.parameters[i])
            push!(expr.args, :prob)
        end
        if passdata(Φ.parameters[i])
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

function initialdata(zp::ExtendedZeroProblem{T}) where T
    ndim = udim(zp)
    u = zeros(T, ndim)
    t = zeros(T, ndim)
    for i in eachindex(zp.u)
        if parent(zp.u[i]) === nothing
            u[zp.ui[i]] .= zp.u[i].u0
            t[zp.ui[i]] .= zp.u[i].t0
        end
    end
    data = ((initialdata(ϕ).data for ϕ in zp.ϕ)...,)
    return (u=u, TS=t, data=data)
end

getvar(zp::ExtendedZeroProblem, u::Symbol) = getvar(getvarinfo(zp, u))
getvar(zp::ExtendedZeroProblem, u::Var) = u
getvar(prob::AbstractContinuationProblem, u) = getvar(getzeroproblem(prob), u)

getproblem(zp::ExtendedZeroProblem, f::Symbol) = getproblem(getprobleminfo(zp, f))
getproblem(zp::ExtendedZeroProblem, f::AbstractZeroProblem) = f
getproblem(prob::AbstractContinuationProblem, f) = getproblem(getzeroproblem(prob), f)

function setvaractive!(zp::ExtendedZeroProblem, u::Var, active::Bool)
    setvaractive!(getmfunc(zp, u), active)
    update_ui!(zp)
    return
end
setvaractive!(prob::AbstractContinuationProblem, u, active) = setvaractive!(getzeroproblem(prob), u, active)

end # module
