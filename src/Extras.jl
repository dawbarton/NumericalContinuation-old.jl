module Extras

using ..ZeroProblems: AbstractRegularFunction, ComputedFunction, Var

using LinearAlgebra

export l2norm, pnorm

struct Pnorm{P} <: AbstractRegularFunction end
(::Pnorm{P})(res, u) where P = (res[1] = norm(u, P))

l2norm(u::Var; name) = ComputedFunction(Pnorm{2}(), (u,); name=name, fdim=1)
pnorm(u::Var; name, p=2) = ComputedFunction(Pnorm{p}(), (u,); name=name, fdim=1)

end # module
