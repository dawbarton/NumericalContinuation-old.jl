module NumericalContinuation

# Top-level abstract types
abstract type AbstractContinuationProblem{T} end
abstract type AbstractToolbox{T} end
abstract type AbstractAtlas{T} end

include("utilities.jl")

include("ZeroProblems.jl")
using .ZeroProblems: ExtendedZeroProblem, Var

include("ComputedFunctions.jl")
using .ComputedFunctions: ComputedFunction

include("Coverings.jl")
using .Coverings: Atlas

include("continuationproblems.jl")
include("toolboxes.jl")

include("AlgebraicProblems.jl")


# Core functionality - modules
export Coverings, ZeroProblems

# Core functionality - types
export ContinuationProblem

# Core functionality - functions
export getzeroproblem, getatlas

# Toolboxes
# export AlgebraicProblems

end # module
