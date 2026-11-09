using BioGeoBEARS, Test, CSV, DataFrames

@testset "Multiple-tree fitting and ancestral summaries" begin
    data = joinpath(@__DIR__, "data")
    master = read_tree("((a:1,b:1):1,(c:1,d:1):1);")
    same = read_tree("((b:1,a:1):1,(d:1,c:1):1);")
    alternate = read_tree("((a:1,c:1):1,(b:1,d:1):1);")
    fossil = read_tree("(((a:0.5,x:0.5):0.5,b:1):1,(c:1,d:1):1);")
    ss = StateSpace(["A", "B"]; max_range_size = 2)
    model = Model(:DEC)
    tipdata(tr) = [i == 2 ? 1.0 : 0.0 for _ in 1:ntips(tr), i in 1:nstates(ss)]
    trees = [master, same, alternate, fossil]
    fits = fit_multiple_trees(trees, ss, [tipdata(tr) for tr in trees], model;
                              algorithms = (:LN_BOBYQA,), maxeval = 30, calc_ancprobs = true)
    @test length(fits) == 4
    @test all(isfinite(f.loglik) for f in fits)
    summary = summarize_tree_ancestry(master, trees, fits)
    @test summary.matches == [4, 4, 4, 4, 4, 3, 3]
    @test all(isapprox(sum(summary.branch_top[n, :]), 1; atol = 1e-8) for n in 1:nnodes(master))
    @test summary.branch_top[6, :] ≈ (fits[1].ancprobs.branch_top[6, :] +
        fits[2].ancprobs.branch_top[6, :] + fits[4].ancprobs.branch_top[7, :]) / 3
    only_alt = summarize_tree_ancestry(master, [alternate], [fits[3]])
    @test only_alt.matches[6:7] == [0, 0]
    @test all(isnan, only_alt.branch_top[6:7, :])
    @test all(isnan, only_alt.branch_bottom[6:7, :])
    @test only_alt.bottom_matches[6:7] == [0, 0]
    @test_throws ArgumentError fit_multiple_trees(BGTree[], ss, [], model)
    @test_throws ArgumentError fit_multiple_trees(trees, ss, [tipdata(master)], model)
    @test_throws ArgumentError summarize_tree_ancestry(master, [read_tree("((a:1,b:1):1,(c:1,x:1):1);")], [fits[1]])
    @test_throws ArgumentError summarize_tree_ancestry(master, [master], [BGBFit(model, -1, 4)])
    strat = Stratification([3.0])
    sfits = fit_multiple_trees([master, same], ss, [tipdata(master), tipdata(same)], model;
                               stratifications = [strat, nothing], algorithms = (:LN_BOBYQA,),
                               maxeval = 30, calc_ancprobs = true)
    @test all(isfinite(f.loglik) for f in sfits)
    @test summarize_tree_ancestry(master, [master, same], sfits).matches == fill(2, nnodes(master))
    @test_throws ArgumentError fit_multiple_trees([master], ss, [tipdata(master)], model; stratifications = [strat, strat])
    # Synthetic probabilities match the input of the R reference generator.
    synthetic = BGBFit[]
    for (i, fit) in enumerate(fits)
        ap = fit.ancprobs
        p = [(n + i) / 30 for n in 1:nnodes(trees[i])]
        top = hcat(p, 1 .- p)
        reference_ap = AncestralProbs(top, top, top, top, ap.downpass)
        push!(synthetic, BGBFit(model, fit.loglik, ntips(trees[i]); ancprobs = reference_ap))
    end
    rref = CSV.read(joinpath(@__DIR__, "baselines", "multiple_trees_r.tsv"), DataFrame; delim = '\t')
    result = summarize_tree_ancestry(master, trees, synthetic; matching = :r)
    @test result.matches == rref.matches
    @test result.branch_top ≈ Matrix(rref[:, 2:3]) atol = 1e-14
    @test_throws ArgumentError summarize_tree_ancestry(master, trees, synthetic; matching = :unknown)
    @test all(isnan, result.branch_bottom[root_node(master), :])
    @test result.bottom_matches[root_node(master)] == 0
end

@testset "Stratified multi-tree fossil fits match R" begin
    data = joinpath(@__DIR__, "data")
    base = joinpath(@__DIR__, "baselines")
    master = read_tree(joinpath(data, "bsm_balanced.newick"))
    trees = [read_tree(joinpath(data, "multitree_fossil$(i).newick")) for i in 1:2]
    tips = read_lagrange_phylip(joinpath(data, "multitree_fossil.data"))
    ss = StateSpace(tips.areanames; max_range_size = 3)
    strat = Stratification(timesfn = joinpath(data, "bsm_small_times.txt"),
                           areas_allowed_fn = joinpath(data, "bsm_small_areas.txt"))
    refs = CSV.read(joinpath(base, "multitree_fossil_strat_fit.tsv"), DataFrame; delim = '\t')
    fitted = fit_multiple_trees(trees, ss, tips, Model(:DEC); stratifications = strat,
                                maxeval = 1000, calc_ancprobs = true)
    fits = BGBFit[]
    for (tr, fit, ref) in zip(trees, fitted, eachrow(refs))
        @test fit.loglik ≈ ref.lnL atol = 1e-5
        @test fit.model["d"] ≈ ref.d atol = 1e-3
        @test fit.model["e"] ≈ ref.e atol = 1e-3
        model = Model(:DEC); model["d"] = ref.d; model["e"] = ref.e
        tc = tip_condlikes(tr, tips; max_range_size = 3)
        ap = stratified_ancestral_probs(tr, ss, tc, model, strat; r_float32 = true)
        @test ap.downpass.loglik ≈ ref.lnL atol = 1e-5
        push!(fits, BGBFit(model, ref.lnL, ntips(tr); ancprobs = ap))
    end
    result = summarize_tree_ancestry(master, trees, fits; matching = :r)
    ref = CSV.read(joinpath(base, "multitree_fossil_strat_summary.tsv"), DataFrame; delim = '\t', missingstring = "NA")
    @test result.matches == ref.matches
    @test result.branch_top ≈ Matrix(ref[:, 2:9]) atol = 1e-5
    for n in 1:nnodes(master)
        if is_root(master, n)
            @test all(isnan, result.branch_bottom[n, :])
        else
            @test result.branch_bottom[n, :] ≈ Float64.(collect(ref[n, 10:17])) atol = 1e-5
        end
    end
end

@testset "Multiple-tree fitted probabilities match R" begin
    data = joinpath(@__DIR__, "data")
    base = joinpath(@__DIR__, "baselines")
    trees = [read_tree(joinpath(data, "bsm_$(name).newick")) for name in ("balanced", "unbalanced")]
    tips = read_lagrange_phylip(joinpath(data, "bsm_small.data"))
    ss = StateSpace(tips.areanames; max_range_size = 3)
    ref = CSV.read(joinpath(base, "multiple_trees_fit_r.tsv"), DataFrame; delim = '\t')
    julia_fits = fit_multiple_trees(trees, ss, tips, Model(:DEC); maxeval = 500, calc_ancprobs = true)
    @test all(f.loglik >= r.lnL - 1e-5 for (f, r) in zip(julia_fits, eachrow(ref)))
    fits = BGBFit[]
    for (tr, row) in zip(trees, eachrow(ref))
        model = Model(:DEC); model["d"] = row.d; model["e"] = row.e
        tc = tip_condlikes(tr, tips; max_range_size = 3)
        ap = ancestral_probs(tr, ss, tc, qmatrix(ss, model; r_float32 = true),
                             cladogenesis_weights(ss, model; r_float32 = true); r_float32 = true)
        @test ap.downpass.loglik ≈ row.lnL atol = 1e-5
        push!(fits, BGBFit(model, ap.downpass.loglik, ntips(tr); ancprobs = ap))
    end
    result = summarize_tree_ancestry(trees[1], trees, fits; matching = :r)
    refsum = CSV.read(joinpath(base, "multiple_trees_fit_summary_r.tsv"), DataFrame; delim = '\t', missingstring = "NA")
    @test result.matches == refsum.matches
    @test result.branch_top ≈ Matrix(refsum[:, 2:9]) atol = 1e-5
    for n in 1:nnodes(trees[1])
        if is_root(trees[1], n)
            @test all(isnan, result.branch_bottom[n, :])
        else
            @test result.branch_bottom[n, :] ≈ Float64.(collect(refsum[n, 10:17])) atol = 1e-5
        end
    end
end
