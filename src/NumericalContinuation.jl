module NumericalContinuation

#-------------------------------------------------------------------------------

"""
    specialize(prob)

Return a specialized problem structure, typically a parameterized structure
for speed. It is assumed that once `specialize` is called, no further changes
to the problem structure are made. `specialize` should not change the problem
in any material way (e.g., the number of equations or variables used).

Note that the resulting specialized problem might share data with the original
problem structure.
"""
function specialize end

specialize(prob) = prob  # default fall-back

#-------------------------------------------------------------------------------

include("ZeroProblems.jl")
include("Coverings.jl")

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
