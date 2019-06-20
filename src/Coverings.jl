"""
	module Coverings


A module that implements advancing local covers from §12.1 of Recipes for
Continuation.
"""
module Coverings

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

function runstatemachine(prob)
    atlas = getatlas(prob)
    state = init_covering!
    while state !== nothing
        state = state(atlas, prob)
    end 
end


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

struct PrCond{T}
    u::Vector{T}
    TS::Vector{T}
end

Base.@kwdef mutable struct AtlasOptions{T}
    # Where possible, use numbers that can be exactly represented with a Float64
    initialstep::T = T(1/2^6)
    initialdirection::Int64 = 1
    stepmin::T = T(1/2^20)
    stepmax::T = T(1)
    stepdecrease::T = T(1/2)
    stepincrease::T = T(1.125)
    cosαmax::T = T(0.99)  # approx 8 degrees
    maxiter::Int64 = 100
end

struct Atlas{T}
    charts::Vector{Chart{T}}
    currentchart::Chart{T}
    prcond::Chart{T}
    currentcurve::Vector{Chart{T}}
    options::AtlasOptions{T}
end

function Atlas(prob)
    # Put the initial guess into a chart structure
    initial = getinitial(getzeroproblem(prob))
    T = eltype(initial.u)
    chart = Chart(pt=0, pt_type=:IP, u=initial.u, TS=initial.TS,
        data=initial.data, R=atlas.options.initialstep, 
        s=atlas.options.initialdirection)
    if initial.correct
        chart.status = :predicted
    else
        chart.status = :corrected
    end
    # Generate an initial projection condition
    prcond = PrCond(copy(initial.u), initial.prcond)
    # Construct the atlas
    return Atlas(Chart{T}[], chart, prcond, AtlasOptions{T}())  # TODO: incorporate user provided options
end

function init_covering!(atlas::Atlas, prob)
    # Correct the initial point if necessary
    if atlas.currentchart.status === :predicted
        return correct!
    elseif atlas.currentchart.status === :corrected
        return addchart!
    else
        @error "Initial chart has an unknown status" atlas.currentchart
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
