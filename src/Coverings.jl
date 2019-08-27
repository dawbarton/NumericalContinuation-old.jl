"""
	module Coverings


A module that implements advancing local covers from §12.1 of Recipes for
Continuation.
"""
module Coverings

using ..ZeroProblems: Var, ComputedFunction, MonitorFunction, monitorfunction, 
    addfunc!, initialdata_embedded, initialdata_nonembedded, initialvar, 
    uidxrange, fidxrange, udim, fdim, jacobian_ad, getvar, evaluate_embedded!
using ..NumericalContinuation: AbstractContinuationProblem, AbstractAtlas, 
    getoption, getzeroproblem, numtype

using LinearAlgebra

using NLsolve

export Atlas, Chart

#--- Chart

mutable struct Chart{T, DE, DN}
    pt::Int64
    pt_type::Symbol
    ep_flag::Bool
    status::Symbol
    u::Vector{T}
    TS::Vector{T}  # tangent space (not normalized)
    t::Vector{T}  # normalized tangent vector
    s::Int64
    R::T
    data_embed::DE
    data_nonembed::DN
end

# NOTE: for the moment don't specialize on data_embed or data_nonembed; it might not reduce runtime much while increasing compile time
Chart(; pt=-1, pt_type=:unknown, ep_flag=false, status=:new, u, TS, t, s=1, R::T, data_embed=(), data_nonembed=()) where T = 
    Chart(pt, pt_type, ep_flag, status, u, TS, t, s, R, data_embed, data_nonembed)

#--- Projection condition (pseudo-arclength equation)

struct PrCond{T}
    u::Vector{T}
    TS::Vector{T}
end

function PrCond(prob::AbstractContinuationProblem)
    T = numtype(prob)
    n = udim(prob)
    return PrCond{T}(zeros(T, n), zeros(T, n))
end

function (prcond::PrCond{T})(u) where T
    res = zero(T)
    for i in eachindex(prcond.u)
        res += prcond.TS[i]*(u[i] - prcond.u[i])
    end
    return res
end

function initial_prcond!(prcond::PrCond{T}, chart::Chart, contvar::Var) where T
    prcond.u .= chart.u
    prcond.TS .= zero(T)
    prcond.TS[uidxrange(contvar)] .= one(T)
    return
end

function update_prcond!(prcond::PrCond, chart::Chart)
    prcond.u .= chart.u
    prcond.TS .= chart.TS
    return
end

#--- AtlasOptions

mutable struct AtlasOptions{T}
    correctinitial::Bool
    initialstep::T
    initialdirection::Int64
    stepmin::T
    stepmax::T
    stepdecrease::T
    stepincrease::T
    αmax::T
    ga::T
    maxiter::Int64
    prcond::Any
end

function AtlasOptions(prob::AbstractContinuationProblem)
    T = numtype(prob)
    # Where possible, use numbers that can be exactly represented with a Float64
    correctinitial = getoption(prob, :atlas, :correctinitial, default=true)
    initialstep = getoption(prob, :atlas, :initialstep, default=T(1/2^6))
    initialdirection = getoption(prob, :atlas, :initialdirection, default=1)
    stepmin = getoption(prob, :atlas, :stepmin, default=T(1/2^20))
    stepmax = getoption(prob, :atlas, :stepmax, default=T(1))
    stepdecrease = getoption(prob, :atlas, :stepdecrease, default=T(1/2))
    stepincrease = getoption(prob, :atlas, :stepincrease, default=T(1.125))
    αmax = getoption(prob, :atlas, :αmax, default=T(0.125))  # approx 7 degrees
    ga = getoption(prob, :atlas, :ga, default=T(0.95))  # adaptation security factor
    maxiter = getoption(prob, :atlas, :maxiter, default=100)
    prcond = getoption(prob, :atlas, :prcond, default=PrCond)
    return AtlasOptions(correctinitial, initialstep, initialdirection, stepmin, 
        stepmax, stepdecrease, stepincrease, αmax, ga, maxiter, prcond)
end

#--- Atlas

"""
    Atlas

# Options

$(fieldnames(AtlasOptions))
"""
mutable struct Atlas{T, C, P} <: AbstractAtlas
    charts::Vector{C}
    currentchart::C
    prcond::P
    prcondzp::ComputedFunction{T, MonitorFunction{T, P}}
    currentcurve::Vector{C}
    options::AtlasOptions{T}
    contvar::Var{T}
end

function Atlas(prob::AbstractContinuationProblem, contvar::Var{T}) where T
    # Set up the options
    options = AtlasOptions(prob)
    # Add the projection condition to the zero problem
    prcond = options.prcond(prob)
    prcondzp = addfunc!(prob, monitorfunction(prcond, getvar(prob, :allvars), name=:prcond, initialvalue=0))
    # Check dimensionality
    zp = getzeroproblem(prob)
    n = udim(zp)
    if n != fdim(zp)
        throw(ErrorException("Dimension mismatch; expected number of equations to match number of continuation variables"))
    end
    # Put the initial guess into a chart structure
    iu = initialvar(zp)
    id_embed = initialdata_embedded(zp)
    id_nonembed = initialdata_nonembedded(zp)
    @assert length(iu.u) == n
    currentchart = Chart(pt=0, pt_type=:IP, u=iu.u, TS=iu.TS, t=zeros(T, n),
        data_embed=id_embed, data_nonembed=id_nonembed, R=options.initialstep, 
        s=options.initialdirection)
    currentchart.t .= currentchart.TS.*currentchart.s
    normTS = norm(iu.TS)
    if normTS > 0
        initial.t ./= normTS
    end
    # Determine the first state
    if options.correctinitial
        currentchart.status = :predicted
    else
        currentchart.status = :corrected
    end
    # Other variables
    charts = Vector{typeof(currentchart)}()
    currentcurve = Vector{typeof(currentchart)}()
    return Atlas(charts, currentchart, prcond, prcondzp, currentcurve, options, contvar)
end

getcontvar(atlas::Atlas) = atlas.contvar

function (atlas::Atlas)(prob::AbstractContinuationProblem)
    # Finite state machine to do the continuation
    state = Base.RefValue{Any}(init_covering!)
    while state[] !== nothing
        state[](atlas, prob, state)
    end 
    return prob
end

#--- Finite state machine states for the atlas

"""
    init_covering!(atlas, prob, nextstate)

Initialise the data structures associated with the covering (atlas) algorithm.

# Outline

1. Determine the initial projection condition.
2. Set the chart status to be
    * `:predicted` if the initial solution should be corrected (default), or
    * `:corrected` if the initial solution should not be modified.

# Next state

* [`Coverings.correct!`](@ref) if chart status is `:predicted`; otherwise
* [`Coverings.addchart!`](@ref).
"""
function init_covering!(atlas::Atlas, prob, nextstate)
    # Set up the initial projection condition
    initial_prcond!(atlas.prcond, atlas.currentchart, atlas.contvar)
    # Choose the next state
    if atlas.currentchart.status === :predicted
        nextstate[] = correct!
    elseif atlas.currentchart.status === :corrected
        nextstate[] = addchart!
    else
        throw(ErrorException("currentchart has an invalid initial status"))
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
    sol = nlsolve((res, u) -> evaluate_embedded!(res, zp, u, prob, chart.data_embed), chart.u)
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
    dfdu = jacobian_ad(zp, chart.u, prob, chart.data_embed)
    dfdp = zeros(T, length(chart.u))
    dfdp[fidxrange(atlas.prcondzp)] .= one(T)
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
    update_prcond!(atlas.prcond, predicted)
    nextstate[] = correct!
    return
end

end # module
