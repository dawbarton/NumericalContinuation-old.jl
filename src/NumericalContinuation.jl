module NumericalContinuation

include("utilities.jl")

include("continuationproblems.jl")

include("ZeroProblems.jl")
include("Coverings.jl")

include("AlgebraicProblems.jl")

using .Coverings: Atlas
using .ZeroProblems: ZeroProblem

# Core functionality - modules
export Coverings, ZeroProblems

# Core functionality - types
export ContinuationProblem

# Core functionality - functions
export getzeroproblem, getatlas

# Toolboxes
export AlgebraicProblems

end # module
