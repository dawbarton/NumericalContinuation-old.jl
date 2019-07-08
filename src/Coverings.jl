"""
	module Coverings


A module that implements advancing local covers from §12.1 of Recipes for
Continuation.
"""
module Coverings

using ..ZeroProblems: AbstractZeroSubproblem, getinitial, fidx, udim, fdim
using ..NumericalContinuation: getoption, getzeroproblem
import ..ZeroProblems: residual!
import ..NumericalContinuation: specialize, setuseroptions!

export Atlas, Chart

# Flow:
#   • init_covering!
#     • Create a CurveSegment with associated projection condition
#     • Enter the FSM loop at the correct! function
#   • correct!
#     • Solve the zero-problem
#     • If error, either quit or refine (if possible)
#   • add!
#     • Add the chart to the CurveSegment
#       • Update the tangent
#       • Update monitor functions
#       • Locate events
#   • flush!
#     • Add the charts in the CurveSegment to the atlas
#     • Clear the CurveSegment
#   • predict!
#     • 

#-------------------------------------------------------------------------------



# function addchart!(cseg::CurveSegment{C}, chart::C, prob) where C
#     if chart.status !== :corrected
#         @error "Uncorrected chart added to curve segment" chart
#     end
#     # Calculate the tangent vector
#     zp = getzeroproblem(prob)
#     dfdu = jacobian(zp, prob)
#     dfdp = zeros(eltype(dfdu), size(dfdu, 1))  # TODO: noalloc
#     dfdp[getfidx(prcond, prob)] = one(eltype(dfdu))
#     chart.TS .= dfdu \ dfdp  # TODO: noalloc
#     # Add the chart to the list of charts
#     push!(cseg, chart)
# end

#-------------------------------------------------------------------------------

struct PrCond{T} <: AbstractZeroSubproblem{T}
    name::Symbol
    deps::Tuple{}
    fdim::Int64
    u::Vector{T}
    TS::Vector{T}
end
PrCond(T::DataType) = PrCond{T}(:prcond, (), 1, Vector{T}(), Vector{T}())

function residual!(res, prcond::PrCond{T}, u) where T
    res[1] = zero(T)
    for i in eachindex(prcond.u)
        res[1] += prcond.TS[i]*(u[i] - prcond.u[i])
    end
    return res
end

#-------------------------------------------------------------------------------

# Chart contains atlas algorithm specific data
Base.@kwdef mutable struct Chart{T, D}
    pt::Int64 = -1
    pt_type::Symbol = :unknown
    ep_flag::Bool = false
    status::Symbol = :new
    u::Vector{T}
    TS::Vector{T}
    s::Int64 = 1
    R::T
    data::D = nothing
end
Chart(T::DataType) = Chart{T, Any}(u=Vector{T}(), TS=Vector{T}(), R=zero(T))

specialize(chart::Chart) = Chart(pt=chart.pt, pt_type=chart.pt_type, ep_flag=chart.ep_flag, 
    status=chart.status, u=chart.u, TS=chart.TS, s=chart.s, R=chart.R, data=(chart.data...,)) 

#-------------------------------------------------------------------------------

Base.@kwdef mutable struct AtlasOptions{T}
    # Where possible, use numbers that can be exactly represented with a Float64
    correctinitial::Bool = true
    initialstep::T = T(1/2^6)
    initialdirection::Int64 = 1
    stepmin::T = T(1/2^20)
    stepmax::T = T(1)
    stepdecrease::T = T(1/2)
    stepincrease::T = T(1.125)
    cosαmax::T = T(0.99)  # approx 8 degrees
    maxiter::Int64 = 100
end
AtlasOptions(T::DataType) = AtlasOptions{T}()

#-------------------------------------------------------------------------------

mutable struct Atlas{T, D}
    charts::Vector{Chart{T, D}}
    currentchart::Chart{T, D}
    prcond::PrCond{T}
    prcondidx::Base.RefValue{Int64}
    currentcurve::Vector{Chart{T, D}}
    options::AtlasOptions{T}
end

function Atlas(T::DataType)
    D = Any
    charts = Vector{Chart{T, D}}()
    currentchart = Chart(T)
    prcond = PrCond(T)
    prcondidx = Ref(zero(Int64))
    currentcurve = Vector{Chart{T, D}}()
    options = AtlasOptions(T)
    return Atlas{T, D}(charts, currentchart, prcond, prcondidx, currentcurve, options)
end

function specialize(atlas::Atlas)
    # Specialize based on the currentchart
    currentchart = specialize(atlas.currentchart)
    C = typeof(currentchart)
    charts = convert(Vector{C}, atlas.charts)
    currentcurve = convert(Vector{C}, atlas.currentcurve)
    return Atlas(charts, currentchart, atlas.prcond, currentcurve, atlas.options)
end

function setuseroptions!(atlas::Atlas, options::Dict)
    optfields = Set(fieldnames(AtlasOptions))
    for opt in options
        if opt[1] in optfields
            setfield!(atlas.options, opt[1], opt[2])
        else
            @info "Unused atlas option" opt
        end
    end
    return atlas
end

#-------------------------------------------------------------------------------

function runstatemachine(prob)
    state = init_covering!(getatlas(prob), prob)
    if getoption(prob, :general, :specialize, default=true)
        _prob = specialize(prob)
    else
        _prob = prob
    end
    return _runstatemachine(getatlas(_prob), _prob, state)
end

@noinline function _runstatemachine(atlas::Atlas, prob, state)  # a function barrier
    while state !== nothing
        state = state(atlas, prob)
    end 
    return prob
end

function init_covering!(atlas::Atlas{T}, prob) where T
    # Add the projection condition to the zero problem
    zp = getzeroproblem(prob)
    push!(zp, atlas.prcond)
    atlas.prcondidx[] = fidx(zp, atlas.prcond)  # store the location within the problem structure
    # Check dimensionality
    n = udim(zp)
    if n != fdim(zp)
        throw(ErrorException("Dimension mismatch; expected number of equations to match number of continuation variables"))
    end
    # Put the initial guess into a chart structure
    initial = getinitial(zp)
    @assert length(initial.u) == n
    atlas.currentchart = Chart{T, Any}(pt=0, pt_type=:IP, u=initial.u, TS=initial.TS, 
        data=initial.data, R=atlas.options.initialstep, s=atlas.options.initialdirection)
    # Set up the initial projection condition (TODO: this could be generalised for other projection conditions)
    resize!(atlas.prcond.u, n) .= initial.u
    resize!(atlas.prcond.TS, n) .= zero(T)
    atlas.prcond.TS[end] = one(T)  # FIXME: assume the last variable is the continuation variable
    # Determine the first state
    if atlas.options.correctinitial
        atlas.currentchart.status = :predicted
        return correct!
    else
        atlas.currentchart.status = :corrected
        return addchart!
    end
end

function correct!(atlas::Atlas, prob)
    # Solve zero problem
    zp = getzeroproblem(prob)
    sol = nlsolve(zp, atlas.currentchart.u)
    if converged(sol)
        atlas.currentchart.u .= sol.zero
        atlas.currentchart.status = :corrected
        return addchart!
    else
        atlas.currentchart.status = :rejected
        return refine!
    end
end

function addchart!(atlas::Atlas, prob)
    chart = atlas.currentchart
    @assert chart.status === :corrected "Chart has not been corrected before adding"
    if chart.pt >= atlas.options.maxiter
        chart.pt_type = :EP
        chart.ep_flag = true
    end
    # TODO: check for the angle
    # TODO: update the tangent direction
    @error "Tangent direction not updated!"
    push!(atlas.currentcurve, chart)
    return flush!
end

function refine!(atlas::Atlas, prob)
    if isempty(atlas.currentcurve)
        return flush!
    else
        chart = first(atlas.currentcurve)
        if chart.R > atlas.options.stepmin
            chart.R = max(chart.R*atlas.options.stepdecrease, atlas.options.stepmin)
            return predict!
        else
            return flush!
        end
    end
end

function flush!(atlas::Atlas, prob)
    added = false
    for chart in atlas.currentcurve
        # Flush any corrected points
        # TODO: check for end points?
        if chart.status === :corrected
            chart.status = :flushed
            push!(atlas.charts, chart)
            added = true
        end
    end
    if added
        # Set the new base point to be the last point flushed
        resize!(atlas.currentcurve, 1)
        atlas.currentcurve[1] = last(atlas.charts)
        return predict!
    else
        # Nothing was added so the continuation failed
        # TODO: indicate the type of failure?
        return nothing
    end
end

function predict!(atlas::Atlas, prob)
    @assert length(atlas.currentcurve) == 1 "Multiple charts in atlas.currentcurve"
    # Copy the existing chart along with toolbox data
    predicted = deepcopy(first(atlas.currentcurve))
    # Predict
    predicted.u .+= predicted.R*predicted.TS*predicted.s
    predicted.status = :predicted
    atlas.currentchart = predicted
    return correct!
end


end # module
