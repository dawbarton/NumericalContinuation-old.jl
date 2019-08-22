using Test

using NumericalContinuation
using NumericalContinuation.ZeroProblems
using NumericalContinuation.Coverings
using NumericalContinuation.AlgebraicProblems

using NLsolve

@testset "ComputedFunction specification (cubic)" begin
    res = zeros(1)
    f = u -> u[1]^3 - u[2]
    sp = ComputedFunction(f, [1.0, 1.0], t0=[0, 0])
    residual!(res, sp, [1.5, 1.0])
    @test res[1] ≈ 2.375
    f! = (res, u) -> (res[1] = u[1]^3 - u[2])
    @test_throws ArgumentError ComputedFunction(f!, [1.0, 1.0], inplace=true)
    sp2 = ComputedFunction(f!, [1.0, 1.0], fdim=1, inplace=true)
    res[1] = 0.0
    residual!(res, sp, [1.5, 1.0])
    @test res[1] ≈ 2.375
    @test_throws ArgumentError ComputedFunction(f!, [1.0, 1.0], t0=[1], fdim=1, inplace=true)
end

@testset "Algebraic continuation (cubic)" begin
    prob = ContinuationProblem()
    f = (u, p) -> u^3 - p
    AlgebraicProblem!(prob, f, 1.5, 1.0, pnames=[:μ])
    res = zeros(fdim(prob))
    data = ZeroProblems.initialdata(getzeroproblem(prob))
    residual!(res, prob, data.u, prob, data.data)
    @test res ≈ [1.5^3-1.0, 0.0]
end

@testset "Cylinder/plane intersection" begin
    prob = ContinuationProblem()
    circle = ComputedFunction!(prob, u -> u[1]^2 + u[2]^2 - 1, [1, 0])
    @test_throws ArgumentError plane = ComputedFunction((u, z) -> u[1] + u[2] + z[1], (circle, [-1]))
    plane = ComputedFunction!(prob, (u, z) -> u[1] + u[2] + z[1], (circle[1], [-1]))
    @test udim(prob) == 3
    @test fdim(prob) == 2
    res0 = zeros(2)
    residual!(res0, prob, [0.1, 0.2, 0.3])
    @test res0 ≈ [-0.95, 0.6]
    solve!(prob, plane[2])
    # prob will now have been specialized; but extra monitor vars have been added so extra data is needed
    res = zeros(fdim(prob))
    residual!(res, prob, [0.1, 0.2, 0.3], prob, prob.atlas.charts[end].data)
    @test res[1:2] ≈ [-0.95, 0.6]
    u = [c.u for c in prob.atlas.charts]
    ux = [u[1] for u in u]
    uy = [u[2] for u in u]
    uz = [u[3] for u in u]
    @test all(@. isapprox(ux^2 + uy^2 - 1, 0, atol=1e-6))
    @test all(@. isapprox(ux + uy + uz, 0, atol=1e-6))
    # Should add some coverage tests - is the manifold properly covered (with 100 points it should be!)
end
