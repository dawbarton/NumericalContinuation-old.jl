using Test

using NumericalContinuation
using NumericalContinuation.ZeroProblems
using NumericalContinuation.AlgebraicProblems

@testset "ZeroSubproblem specification (cubic)" begin
    res = zeros(1)
    f = u -> u[1]^3 - u[2]
    sp = ZeroSubproblem(f, [1.0, 1.0], t0=[0, 0])
    residual!(res, sp, [1.5, 1.0])
    @test res[1] ≈ 2.375
    f! = (res, u) -> (res[1] = u[1]^3 - u[2])
    @test_throws ArgumentError ZeroSubproblem(f!, [1.0, 1.0], inplace=true)
    sp2 = ZeroSubproblem(f!, [1.0, 1.0], fdim=1, inplace=true)
    res[1] = 0.0
    residual!(res, sp, [1.5, 1.0])
    @test res[1] ≈ 2.375
    @test_throws ArgumentError ZeroSubproblem(f!, [1.0, 1.0], t0=[1], fdim=1, inplace=true)
end

@testset "Algebraic continuation (cubic)" begin
    res = zeros(1)
    f = (u, p) -> u^3 - p
    ap = AlgebraicProblem(f, 1.0, 1.0)
    residual!(res, ap, [1.5], [1.0])
    @test res[1] ≈ 2.375
end

@testset "Cylinder/plane intersection" begin
    circle = ZeroSubproblem(u -> u[1]^2 + u[2]^2 - 1, [1, 0])
    @test_throws ArgumentError plane = ZeroSubproblem((u, z) -> u[1] + u[2] + z[1], (circle, [-1]))
    plane = ZeroSubproblem((u, z) -> u[1] + u[2] + z[1], (circle[1], [-1]))
    prob = ContinuationProblem()
    push!(prob, circle)
    push!(prob, plane)
    @test udim(prob) == 3
    @test fdim(prob) == 2
    res0 = zeros(2)
    residual!(res0, prob, [0.1, 0.2, 0.3])
    @test res0 ≈ [-0.95, 0.6]
    probz = NumericalContinuation.specialize(prob)
    res = zeros(2)
    residual!(res, probz, [0.1, 0.2, 0.3])
    @test res ≈ [-0.95, 0.6]
    # prob = NumericalContinuation.ContinuationProblem()
    # push!(prob, circle)
    # push!(prob, plane)
    # NumericalContinuation.Coverings.init_covering!(prob.atlas, prob)
end