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
    evaluate_embedded!(res, zp, u, prob, data)
    @test res ≈ [1.5^3-1.0, 0.0]
end

@testset "Cylinder/plane intersection" begin
    createprob = () -> begin
            prob = ContinuationProblem()
            add!(prob, zeroproblem((res, u) -> res[1] = u[1]^2 + u[2]^2 - 1, [1, 0], name=:circle, fdim=1))
            add!(prob, zeroproblem((res, u, z) -> res[1] = u[1] + u[2] + z[1], (getvar(prob, :circle_u1), [-1]), name=:plane, fdim=1))
            return prob
        end
    prob = createprob()
    @test_throws ArgumentError plane = zeroproblem((res, u, z) -> res[1] = u[1] + u[2] + z[1], (prob, [-1]), name=:plane, fdim=1)
    @test udim(prob) == 3
    @test fdim(prob) == 2
    res0 = zeros(2)
    zp = getzeroproblem(prob)
    ZeroProblems.evaluate_embedded!(res0, zp, [0.1, 0.2, 0.3])
    @test res0 ≈ [-0.95, 0.6]
    solve!(prob, :plane_u2)
    # prob will now have been specialized; but extra monitor vars have been added so extra data is needed
    res = zeros(fdim(prob))
    zp = getzeroproblem(prob)  # zp might have changed during the solve!
    ZeroProblems.evaluate_embedded!(res, zp, [0.1, 0.2, 0.3], prob, prob.atlas.charts[end].data_embed)
    @test res[1:2] ≈ [-0.95, 0.6]
    u = [c.u for c in prob.atlas.charts]
    ux = [u[1] for u in u]
    uy = [u[2] for u in u]
    uz = [u[3] for u in u]
    @test all(@. isapprox(ux^2 + uy^2 - 1, 0, atol=1e-6))  # on the manifold (1)
    @test all(@. isapprox(ux + uy + uz, 0, atol=1e-6))  # on the manifold (2)
    @test maximum(uz) > 1.3 && minimum(uz) < -1.3  # coverage
    # try to break it! should cause the refine code to trigger
    prob = createprob()
    setoption!(prob, :atlas, :stepmax, 10)
    setoption!(prob, :atlas, :αmaz, π/2)
    setoption!(prob, :atlas, :initialstep, 10)
    solve!(prob, :plane_u2)
    u = [c.u for c in prob.atlas.charts]
    ux = [u[1] for u in u]
    uy = [u[2] for u in u]
    uz = [u[3] for u in u]
    @test all(@. isapprox(ux^2 + uy^2 - 1, 0, atol=1e-6))  # on the manifold (1)
    @test all(@. isapprox(ux + uy + uz, 0, atol=1e-6))  # on the manifold (2)
end

@testset "ZeroProblems coverage" begin
    @test_throws ArgumentError Var(:test, 1, u0='a')
    u = Var(:a, 1, T=Float64)
    @test !isempty(string(u))  # test show doesn't throw
    prob = ContinuationProblem()
    func = add!(prob, zeroproblem((res, u) -> res[1] = u[1]^2 + u[2]^2 - 1, [1, 0], name=:circle, fdim=1))
    @test !isempty(string(prob))
    @test nameof(func[:circle_u1]) === :circle_u1
    @test getvars(prob) == [:allvars, :circle_u1]
    @test getvar(prob, func[:circle_u1]) === func[:circle_u1]
    @test getfuncs(prob) == [:circle]
    @test getfunc(prob, :circle) === func
    @test getfunc(prob, func) === func
    @test hasfunc(prob, func)
    @test hasfunc(prob, :circle)
    @test !hasfunc(prob, :notcircle)
    @test hasvar(prob, func[:circle_u1])
    @test hasvar(prob, :circle_u1)
    @test !hasvar(prob, :random)
end
