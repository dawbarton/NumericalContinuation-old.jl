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
    data::D = ()
end
Chart(T::DataType) = Chart{T, Any}(u=Vector{T}(), TS=Vector{T}(), R=zero(T))

# specialize(chart::Chart) = chart
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
    prcondidx::Int64
    currentcurve::Vector{Chart{T, D}}
    options::AtlasOptions{T}
end

function Atlas(T::DataType)
    D = Any
    charts = Vector{Chart{T, D}}()
    currentchart = Chart(T)
    prcond = PrCond(T)
    prcondidx = 0
    currentcurve = Vector{Chart{T, D}}()
    options = AtlasOptions(T)
    return Atlas{T, D}(charts, currentchart, prcond, prcondidx, currentcurve, options)
end

function specialize(atlas::Atlas)
    # Specialize based on the current chart
    currentchart = specialize(atlas.currentchart)
    C = typeof(currentchart)
    charts = convert(Vector{C}, atlas.charts)
    currentcurve = convert(Vector{C}, atlas.currentcurve)
    return Atlas(charts, currentchart, atlas.prcond, atlas.prcondidx, currentcurve, atlas.options)
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

"""
    init_covering!(atlas, prob)

Initialise the data structures associated with the covering (atlas) algorithm.

# Outline

1. Add the projection condition to the zero problem.
2. Get the initial solution and put into a chart structure.
3. Determine the initial projection condition.
4. Set the chart status to be
    * `:predicted` if the initial solution should be corrected (default), or
    * `:corrected` if the initial solution should not be modified.

# Next state

* [`Coverings.correct!`](@ref) if chart status is `:predicted`; otherwise
* [`Coverings.addchart!`](@ref).
"""
function init_covering!(atlas::Atlas{T}, prob) where T
    # Add the projection condition to the zero problem
    zp = getzeroproblem(prob)
    push!(zp, atlas.prcond)
    atlas.prcondidx = fidx(zp, atlas.prcond)  # store the location within the problem structure
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

"""
    correct!(atlas, prob)

Correct the (predicted) solution in the current chart with the projection
condition as previously specified.

# Outline

1. Solve the zero-problem with the current chart as the starting guess.
2. Determine whether the solver converged;
    * if converged, set the chart status to `:corrected`, otherwise
    * if not converged, set the chart status to `:rejected`.

# Next state

* [`Coverings.addchart!`](@ref) if the chart status is `:corrected`; otherwise
* [`Coverings.refine!`](@ref).
"""
function correct!(atlas::Atlas, prob)
    # Solve zero problem
    zp = getzeroproblem(prob)
    chart = atlas.currentchart
    sol = nlsolve((res, u) -> residual!(res, zp, u, prob, chart.data), chart.u)
    if converged(sol)
        chart.u .= sol.zero
        chart.status = :corrected
        return addchart!
    else
        chart.status = :rejected
        return refine!
    end
end

"""
    addchart!(atlas, prob)

Add a corrected chart to the list of charts that defines the current curve and
update any calculated properties (e.g., tangent vector).

# Outline

1. Determine whether the chart is an end point (e.g., the maximum number of
   iterations has been reached).
2. Update the tangent vector of the chart.
3. Check whether the chart provides an adequate representation of the current
   curve (e.g., whether the angle between the tangent vectors is sufficiently
   small).

# Next state

* [`Coverings.flush!`](@ref).

# To do

1. Update monitor functions.
2. Locate events.
"""
function addchart!(atlas::Atlas{T}, prob) where T
    zp = getzeroproblem(prob)
    chart = atlas.currentchart
    @assert chart.status === :corrected "Chart has not been corrected before adding"
    if chart.pt >= atlas.options.maxiter
        chart.pt_type = :EP
        chart.ep_flag = true
    end
    dfdu = ZeroProblems.jacobian_ad(zp, chart.u, prob, chart.data)
    dfdp = zeros(T, length(chart.u))
    dfdp[fidx(atlas.prcondidx)] = one(T)
    chart.TS .= dfdu \ dfdp
    # TODO: check for the angle
    push!(atlas.currentcurve, chart)
    return flush!
end

"""
    refine!(atlas, prob)

Update the continuation strategy to attempt to progress the continuation after
a failed correction step.

# Outline

1. If the step size is greater than the minimum, reduce the step size to the
   larger of the minimum step size and the current step size multiplied by the
   step decrease factor.

# Next state

* [`Coverings.predict!`](@ref) if the continuation strategy was updated,
  otherwise
* [`Coverings.flush!`](@ref).
"""
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

"""
    flush!(atlas, prob)

Given a representation of a curve in the form of a list of charts, add all
corrected charts to the atlas, and update the current curve.

# Outline

1. Add all corrected charts to the atlas.
2. If charts were added to the atlas, set the current curve to be a single
   chart at the boundary of the atlas.
   
# Next state

* [`Coverings.predict!`](@ref) if charts were added to the atlas, otherwise
* `nothing` to terminate the state machine.
"""
function flush!(atlas::Atlas, prob)
    added = false
    ep_flag = false
    for chart in atlas.currentcurve
        # Flush any corrected points
        # TODO: check for end points?
        if chart.status === :corrected
            chart.status = :flushed
            push!(atlas.charts, chart)
            added = true
            ep_flag |= chart.ep_flag
        end
    end
    if added
        # Set the new base point to be the last point flushed
        resize!(atlas.currentcurve, 1)
        atlas.currentcurve[1] = last(atlas.charts)
        if ep_flag
            return nothing
        else
            return predict!
        end
    else
        # Nothing was added so the continuation failed
        # TODO: indicate the type of failure?
        return nothing
    end
end

"""
    predict!(atlas, prob)

Make a deep copy of the (single) chart in the current curve and generate a
prediction for the next chart along the curve.

# Outline

1. `deepcopy` the chart in the current curve.
2. Generate a predicted value for the solution.
3. Set the current chart equal to the predicted value.
4. Update the projection condition with the new prediction and tangent vector.

# Next state

* [`Coverings.correct!`](@ref).
"""
function predict!(atlas::Atlas, prob)
    @assert length(atlas.currentcurve) == 1 "Multiple charts in atlas.currentcurve"
    # Copy the existing chart along with toolbox data
    predicted = deepcopy(first(atlas.currentcurve))
    # Predict
    predicted.pt += 1
    predicted.u .+= predicted.R*predicted.TS*predicted.s
    predicted.status = :predicted
    atlas.currentchart = predicted
    # Update the projection condition
    prcond.u .= predicted.u
    prcond.TS .= predicted.TS
    return correct!
end


end # module
