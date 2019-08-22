module ComputedFunctions

using ..ZeroProblem: ZeroProblem, Var, initialdata
using ..NumericalContinuation: AbstractContinuationProblem

#--- ComputedFunction

"""
    ComputedFunction{T, F}
"""
mutable struct ComputedFunction{T, F}
    name::Symbol
    deps::Vector{Var{T}}
    f!::F
    fdim::Int64
    vars::Dict{Symbol, Var{T}}
    idx::Int64
    idxrange::UnitRange{Int64}
end

function ComputedFunction(f, u0::NTuple{N, Var{T}}; fdim=0, name=:computed, inplace=false) where {N, T}
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
    # 
    return ComputedFunction(name, deps, f!, fdim, vars, 0, 0:0)
end

Base.nameof(prob::ComputedFunction) = prob.name
ZeroProblem.dependencies(prob::ComputedFunction) = prob.deps
ZeroProblem.fdim(prob::ComputedFunction) = prob.fdim
ZeroProblem.numtype(prob::ComputedFunction{T}) where T = T
Base.getindex(prob::ComputedFunction, idx::Integer) = getindex(prob.deps, idx)
Base.getindex(prob::ComputedFunction, sym::Symbol) = prob.vars[sym]
ZeroProblem.fidx(prob::ComputedFunction) = prob.idx
ZeroProblem.fidxrange(prob::ComputedFunction) = prob.idxrange
ZeroProblem.getfunc(prob::ComputedFunction) = prob.f!

ZeroProblem.passdata(::Type{<: ComputedFunction{T, F}}) where {T, F} = passdata(F)
ZeroProblem.passproblem(::Type{<: ComputedFunction{T, F}}) where {T, F} = passproblem(F)

#--- ExtendedComputedFunctions - all the computed functions

struct ExtendedComputedFunctions{T, D, Φ}
    u::Vector{Var{T}}
    udim::Base.RefValue{Int64}
    usym::Dict{Symbol, Var{T}}
    ϕ::Φ
    ϕdeps::Vector{Tuple{Vararg{Int64, N} where N}}
    ϕdim::Base.RefValue{Int64}
    ϕsym::Dict{Symbol, ZeroProblem{T}}
end

function ExtendedComputedFunctions(T=Float64) 
    prob = ExtendedComputedFunctions{T, Nothing, Vector{ZeroProblem{T}}}(
        Vector{Var{T}}(),                               # u
        Ref(zero(Int64)),                               # udim
        Dict{Symbol, Var{T}}(),                         # usym
        Vector{ZeroProblem{T}}(),                       # ϕ
        Vector{Tuple{Vararg{Int64, N} where N}}(),      # ϕdeps
        Ref(zero(Int64)),                               # ϕdim
        Dict{Symbol, ZeroProblem{T}}(),                 # ϕsym
    )
    return prob
end


end # module
