"""
	module Coverings


A module that implements advancing local covers from §12.1 of Recipes for
Continuation.
"""
module Coverings

using ..ZeroProblems: monitorfunction, initialdata, uidx, uidxrange, fidx, fidxrange, 
    udim, fdim, jacobian_ad, Var, getvar
using ..NumericalContinuation: AbstractAtlas, getoption, getzeroproblem, getatlas
import ..ZeroProblems: residual!
import ..NumericalContinuation: specialize, setuseroptions!

using LinearAlgebra

using NLsolve

export Atlas, Chart

#-------------------------------------------------------------------------------

struct PrCond{T}
    u::Vector{T}
    TS::Vector{T}
end
PrCond(T::Type) = PrCond{T}(Vector{T}(), Vector{T}())

function (prcond::PrCond{T})(u) where T
    res = zero(T)
    for i in eachindex(prcond.u)
        res += prcond.TS[i]*(u[i] - prcond.u[i])
    end
    return res
end

#-------------------------------------------------------------------------------

# Chart contains atlas algorithm specific data
Base.@kwdef mutable struct Chart{T, D <: Tuple}
    pt::Int64 = -1
    pt_type::Symbol = :unknown
    ep_flag::Bool = false
    status::Symbol = :new
    u::Vector{T}
    TS::Vector{T}  # tangent space (not normalized)
    t::Vector{T}  # normalized tangent vector
    s::Int64 = 1
    R::T
    data::D = ()
end
Chart(T::Type) = Chart{T, Tuple}(u=Vector{T}(), TS=Vector{T}(), t=Vector{T}(), R=zero(T))

# specialize(chart::Chart) = chart
specialize(chart::Chart) = Chart(pt=chart.pt, pt_type=chart.pt_type, ep_flag=chart.ep_flag, 
    status=chart.status, u=chart.u, TS=chart.TS, t=chart.t, s=chart.s, R=chart.R, data=(chart.data...,)) 

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
    αmax::T = T(0.125)  # approx 7 degrees
    ga::T = T(0.95)  # adaptation security factor
    maxiter::Int64 = 100
end
AtlasOptions(T::Type) = AtlasOptions{T}()

#-------------------------------------------------------------------------------

mutable struct Atlas{T, D} <: AbstractAtlas{T}
    charts::Vector{Chart{T, D}}
    currentchart::Chart{T, D}
    prcond::PrCond{T}
    prcondidx::Int64
    contvar::Var{T}
    contvaridx::Int64
    currentcurve::Vector{Chart{T, D}}
    options::AtlasOptions{T}
end

function Atlas(T::Type)
    D = Tuple
    charts = Vector{Chart{T, D}}()
    currentchart = Chart(T)
    prcond = PrCond(T)
    prcondidx = 0
    contvar = Var(:null, 1, T=T)
    contvaridx = 0
    currentcurve = Vector{Chart{T, D}}()
    options = AtlasOptions(T)
    return Atlas{T, D}(charts, currentchart, prcond, prcondidx, contvar, contvaridx, currentcurve, options)
end

function specialize(atlas::Atlas)
    # Specialize based on the current chart
    currentchart = specialize(atlas.currentchart)
    C = typeof(currentchart)
    charts = convert(Vector{C}, atlas.charts)
    currentcurve = convert(Vector{C}, atlas.currentcurve)
    return Atlas(charts, currentchart, atlas.prcond, atlas.prcondidx, atlas.contvar, atlas.contvaridx, currentcurve, atlas.options)
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

setcontinuationvar!(atlas::Atlas{T}, contvar::Var{T}) where T = (atlas.contvar = contvar; atlas)

#-------------------------------------------------------------------------------

function runstatemachine!(prob)
    state = Base.RefValue{Any}(nothing)
    init_covering!(getatlas(prob), prob, state)
    if getoption(prob, :general, :specialize, default=true)
        _prob = specialize(prob)
    else
        _prob = prob
    end
    _runstatemachine!(getatlas(_prob), _prob, state)
    return _prob
end

@noinline function _runstatemachine!(atlas::Atlas, prob, state)  # a function barrier
    while state[] !== nothing
        state[](atlas, prob, state)
    end 
    return prob
end

"""
    init_covering!(atlas, prob, nextstate)

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
function init_covering!(atlas::Atlas{T, D}, prob, nextstate) where {T, D}
    # Add the projection condition to the zero problem
    zp = getzeroproblem(prob)
    prcondzp = monitorfunction(atlas.prcond, getvar(prob, :allvars), name=:prcond)
    push!(zp, prcondzp)
    atlas.prcondidx = fidx(prcondzp)  # store the location within the problem structure (TODO: Fix this - should store the ZeroProblem!)
    atlas.contvaridx = uidx(zp, atlas.contvar)
    # Check dimensionality
    n = udim(zp)
    if n != fdim(zp)
        throw(ErrorException("Dimension mismatch; expected number of equations to match number of continuation variables"))
    end
    # Put the initial guess into a chart structure
    initial = initialdata(zp)
    @assert length(initial.u) == n
    atlas.currentchart = Chart{T, D}(pt=0, pt_type=:IP, u=initial.u, TS=initial.TS, t=zeros(T, n),
        data=initial.data, R=atlas.options.initialstep, s=atlas.options.initialdirection)
    atlas.currentchart.t .= atlas.currentchart.TS.*atlas.currentchart.s
    normTS = norm(initial.TS)
    if normTS > 0
        initial.t ./= normTS
    end
    # Set up the initial projection condition (TODO: this could be generalised for other projection conditions)
    resize!(atlas.prcond.u, n) .= initial.u
    resize!(atlas.prcond.TS, n) .= zero(T)
    atlas.prcond.TS[uidxrange(zp, atlas.contvaridx)] .= one(T)
    # Determine the first state
    if atlas.options.correctinitial
        atlas.currentchart.status = :predicted
        nextstate[] = correct!
    else
        atlas.currentchart.status = :corrected
        nextstate[] = addchart!
    end
    return
end

"""
    correct!(atlas, prob, nextstate)

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
function correct!(atlas::Atlas, prob, nextstate)
    # Solve zero problem
    zp = getzeroproblem(prob)
    chart = atlas.currentchart
    sol = nlsolve((res, u) -> residual!(res, zp, u, prob, chart.data), chart.u)
    if converged(sol)
        chart.u .= sol.zero
        chart.status = :corrected
        nextstate[] = addchart!
    else
        chart.status = :rejected
        nextstate[] = refine!
    end
    return
end

"""
    addchart!(atlas, prob, nextstate)

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
function addchart!(atlas::Atlas{T}, prob, nextstate) where T
    zp = getzeroproblem(prob)
    chart = atlas.currentchart
    @assert chart.status === :corrected "Chart has not been corrected before adding"
    if chart.pt >= atlas.options.maxiter
        chart.pt_type = :EP
        chart.ep_flag = true
    end
    # Update the tangent vector
    dfdu = jacobian_ad(zp, chart.u, prob, chart.data)
    dfdp = zeros(T, length(chart.u))
    dfdp[fidxrange(zp, atlas.prcondidx)] .= one(T) # TODO: fix this with a ref to the actual ZeroProblem
    chart.TS .= dfdu \ dfdp
    chart.t .= chart.s.*chart.TS./norm(chart.TS)
    opt = atlas.options
    # Check the angle
    if !isempty(atlas.currentcurve)
        chart0 = atlas.currentcurve[end]
        β = acos(clamp(dot(chart.t, chart0.t), -1, 1))
        if β > opt.αmax*opt.stepincrease
            # Angle is too large, attempt to adjust step size
            if chart0.R > opt.stepmin
                chart.status = :rejected
                chart0.R = clamp(chart0.R*opt.stepdecrease, opt.stepmin, opt.stepmax)
                nextstate[] =  predict!
                return
            else
                @warn "Minimum step size reached but angle constraints not met" chart
            end
        end
        if opt.stepincrease^2*β < opt.αmax
            mult = opt.stepincrease
        else
            mult = clamp(opt.αmax / (sqrt(opt.stepincrease)*β), opt.stepdecrease, opt.stepincrease)
        end
        chart.R = clamp(opt.ga*mult*chart.R, opt.stepmin, opt.stepmax)
    end
    push!(atlas.currentcurve, chart)
    nextstate[] =  flush!
    return
end

"""
    refine!(atlas, prob, nextstate)

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
function refine!(atlas::Atlas, prob, nextstate)
    if isempty(atlas.currentcurve)
        nextstate[] = flush!
    else
        chart = first(atlas.currentcurve)
        if chart.R > atlas.options.stepmin
            chart.R = max(chart.R*atlas.options.stepdecrease, atlas.options.stepmin)
            nextstate[] = predict!
        else
            nextstate[] = flush!
        end
    end
end

"""
    flush!(atlas, prob, nextstate)

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
function flush!(atlas::Atlas, prob, nextstate)
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
            nextstate[] = nothing
        else
            nextstate[] = predict!
        end
    else
        # Nothing was added so the continuation failed
        # TODO: indicate the type of failure?
        nextstate[] = nothing
    end
    return
end

"""
    predict!(atlas, prob, nextstate)

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
function predict!(atlas::Atlas, prob, nextstate)
    @assert length(atlas.currentcurve) == 1 "Multiple charts in atlas.currentcurve"
    # Copy the existing chart along with toolbox data
    predicted = deepcopy(first(atlas.currentcurve))
    # Predict
    predicted.pt += 1
    predicted.u .+= predicted.R*predicted.TS*predicted.s
    predicted.status = :predicted
    atlas.currentchart = predicted
    # Update the projection condition
    atlas.prcond.u .= predicted.u
    atlas.prcond.TS .= predicted.TS
    nextstate[] = correct!
    return
end


end # module
