using Test

using NumericalContinuation
using NumericalContinuation.ZeroProblems
using NumericalContinuation.Coverings
using NumericalContinuation.AlgebraicProblems

using NLsolve

@testset "zeroproblem specification (cubic)" begin
    f = (res, u) -> res[1] = u[1]^3 - u[2]
    @test_throws UndefKeywordError zeroproblem(f, [1.0, 1.0], fdim=1)
    @test_throws UndefKeywordError zeroproblem(f, [1.0, 1.0], name=:sp)
    sp = zeroproblem(f, [1.0, 1.0], t0=[0, 0], fdim=1, name=:sp)
    res = evaluate!(zeros(fdim(sp)), sp, [1.5, 1.0])
    @test res[1] ≈ 2.375
    @test_throws ArgumentError zeroproblem(f, [1.0, 1.0], t0=[1], fdim=1, name=:sp)
end

@testset "Algebraic continuation (cubic)" begin
    prob = ContinuationProblem()
    f = (u, p) -> u^3 - p
    add!(prob, AlgebraicProblem(f, 1.5, 1.0, pnames=[:μ], name=:alg))
    res = zeros(fdim(prob))
    zp = getzeroproblem(prob)
    data = ZeroProblems.initialdata_embedded(zp)
    u = ZeroProblems.initialvar(zp).u
    eval_embedded!(res, zp, u, prob, data)
    @test res ≈ [1.5^3-1.0, 0.0]
end

# @testset "Cylinder/plane intersection" begin
#     prob = ContinuationProblem()
#     circle = add!(prob, zeroproblem(u -> u[1]^2 + u[2]^2 - 1, [1, 0], name=:circle))
#     @test_throws ArgumentError plane = zeroproblem((u, z) -> u[1] + u[2] + z[1], (circle, [-1]), name=:plane)
#     plane = add!(prob, zeroproblem((u, z) -> u[1] + u[2] + z[1], (circle[1], [-1]), name=:plane))
#     @test udim(prob) == 3
#     @test fdim(prob) == 2
#     res0 = zeros(2)
#     evaluate!(res0, prob, [0.1, 0.2, 0.3])
#     @test res0 ≈ [-0.95, 0.6]
#     solve!(prob, plane[2])
#     # prob will now have been specialized; but extra monitor vars have been added so extra data is needed
#     res = zeros(fdim(prob))
#     evaluate!(res, prob, [0.1, 0.2, 0.3], prob, prob.atlas.charts[end].data)
#     @test res[1:2] ≈ [-0.95, 0.6]
#     u = [c.u for c in prob.atlas.charts]
#     ux = [u[1] for u in u]
#     uy = [u[2] for u in u]
#     uz = [u[3] for u in u]
#     @test all(@. isapprox(ux^2 + uy^2 - 1, 0, atol=1e-6))
#     @test all(@. isapprox(ux + uy + uz, 0, atol=1e-6))
#     # Should add some coverage tests - is the manifold properly covered (with 100 points it should be!)
# end
