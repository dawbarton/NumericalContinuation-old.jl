module NumericalContinuation

include("Coverings.jl")
include("ZeroProblems.jl")
include("AlgebraicProblems.jl")

using .Coverings: Atlas
using .ZeroProblems: ZeroProblem

#-------------------------------------------------------------------------------
abstract type AbstractContinuationProblem end

struct ContinuationProblem{T, Z} <: AbstractContinuationProblem
    atlas::Atlas{T}
    zeroproblem::Z
end

getatlas(prob::ContinuationProblem) = prob.atlas
getzeroproblem(prob::ContinuationProblem) = prob.zeroproblem


end # module
