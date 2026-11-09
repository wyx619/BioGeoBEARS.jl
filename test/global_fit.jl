using BioGeoBEARS, Test, Random

@testset "Global search and local refinement" begin
    model = Model(:DEC)
    set_param!(model, "e"; kind = "fixed")
    set_param!(model, "d"; init = 0.2, est = 0.2, lower = 0.01, upper = 5.0)
    objective = (x, _) -> max(1.0 - 4 * (x[1] - 0.2)^2, 2.0 - 10 * (x[1] - 3.2)^2)
    fit = BioGeoBEARS._fit_model(objective, nothing, model, 10;
        algorithms = (:LN_BOBYQA,), maxeval = 100, ftol_rel = 1e-10,
        xtol_rel = 0.0, nstarts = 1, rng = MersenneTwister(1), global_maxeval = 200)
    @test fit.loglik ≈ 2.0 atol = 1e-7
    @test fit.model["d"] ≈ 3.2 atol = 1e-4
    @test model["d"] == 0.2

    data = joinpath(@__DIR__, "data")
    tr = read_tree(joinpath(data, "bsm_balanced.newick"))
    tips = read_lagrange_phylip(joinpath(data, "bsm_small.data"))
    ss = StateSpace(tips.areanames; max_range_size = 3)
    strat = Stratification(timesfn = joinpath(data, "bsm_small_times.txt"),
                           areas_allowed_fn = joinpath(data, "bsm_small_areas.txt"))
    options = (algorithms = (:LN_BOBYQA,), maxeval = 100, calc_ancprobs = true)
    localfit = fit_model(tr, ss, tips, Model(:DEC), strat; options...)
    globalfit = fit_model(tr, ss, tips, Model(:DEC), strat; options..., global_maxeval = 100)
    @test globalfit.loglik >= localfit.loglik - 1e-10
    @test globalfit.ancprobs !== nothing
    @test isfinite(globalfit.loglik)
    @test_throws ArgumentError fit_model(tr, ss, tips, Model(:DEC); global_maxeval = -1)
    @test_throws ArgumentError fit_model(tr, ss, tips, Model(:DEC); maxeval = 0)
    @test isnan(fit_summary(BGBFit(Model(:DEC), -1.0, 3)).AICc)
end
