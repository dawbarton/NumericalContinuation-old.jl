module NumericalContinuation

# Top-level abstract types
abstract type AbstractContinuationProblem{T} end
abstract type AbstractToolbox{T} end
abstract type AbstractAtlas{T} end

include("utilities.jl")

include("ZeroProblems.jl")
using .ZeroProblems

include("Coverings.jl")
using .Coverings

include("continuationproblems.jl")
include("toolboxes.jl")

include("AlgebraicProblems.jl")


# Core functionality - modules
export Coverings, ZeroProblems

# Core functionality - types
export ContinuationProblem

# Core functionality - functions
export getvar, getzeroproblem, getatlas, udim, fdim, solve!, setoption!, 
    getoption, add!

# Toolboxes
export AlgebraicProblems

end # module
