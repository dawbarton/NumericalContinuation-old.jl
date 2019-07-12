module NumericalContinuation

include("utilities.jl")

include("continuationproblems.jl")

include("ZeroProblems.jl")
include("Coverings.jl")

include("AlgebraicProblems.jl")

using .Coverings: Atlas
using .ZeroProblems: ZeroProblem

# Core functionality
export Coverings, ZeroProblems

# Toolboxes
export AlgebraicProblems

end # module
