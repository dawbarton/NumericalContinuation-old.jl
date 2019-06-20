using Documenter, NumericalContinuation

makedocs(
    modules = [NumericalContinuation],
    format = :html,
    checkdocs = :exports,
    sitename = "NumericalContinuation.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/dawbarton/NumericalContinuation.jl.git",
)
