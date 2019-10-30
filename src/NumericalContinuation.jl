# Just a thought...
# """
# $(read(joinpath(@__DIR__, "..", "README.md"), String))
# """
module NumericalContinuation

include("forwarddefinitions.jl")

include("ZeroProblems.jl")
using .ZeroProblems

include("Coverings.jl")
using .Coverings

include("continuationproblems.jl")
include("toolboxes.jl")

include("Extras.jl")
using .Extras

include("AlgebraicProblems.jl")

# Core functionality - modules
export Coverings, ZeroProblems

# Core functionality - types
export ContinuationProblem

# Core functionality - functions
export getvar, getzeroproblem, getatlas, udim, fdim, solve!, setoption!, 
    getoption, add!, getvars, getfuncs, l2norm

# Toolboxes
export AlgebraicProblems

end # module
