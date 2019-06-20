using Test

using NumericalContinuation
using NumericalContinuation.ZeroProblems
using NumericalContinuation.AlgebraicProblems

@testset "ZeroSubproblem specification (cubic)" begin
    res = zeros(1)
    f = u -> u[1]^3 - u[2]
    sp = ZeroSubproblem(f, u0=[1.0, 1.0])
    residual!(res, sp, [1.5, 1.0])
    @test res[1] ≈ 2.375
    f! = (res, u) -> (res[1] = u[1]^3 - u[2])
    @test_throws ArgumentError ZeroSubproblem(f!, u0=[1.0, 1.0])
    sp2 = ZeroSubproblem(f!, u0=[1.0, 1.0], fdim=1)
    res[1] = 0.0
    residual!(res, sp, [1.5, 1.0])
    @test res[1] ≈ 2.375
end

@testset "Algebraic continuation (cubic)" begin
    res = zeros(1)
    f = (u, p) -> u^3 - p
    ap = AlgebraicProblem(f, u0=1.0, p0=1.0)
    residual!(res, ap, [1.5], [1.0])
    @test res[1] ≈ 2.375
end
