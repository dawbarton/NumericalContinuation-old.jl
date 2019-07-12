using Documenter, NumericalContinuation

cd(@__DIR__)

makedocs(
    modules = [NumericalContinuation],
    format = Documenter.HTML(),
    checkdocs = :none, # :exports,
    sitename = "NumericalContinuation.jl",
    pages = Any["index.md"]
)

# deploydocs(
#     repo = "github.com/dawbarton/NumericalContinuation.jl.git",
# )
