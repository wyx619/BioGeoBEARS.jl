using BioGeoBEARS, Test, CSV, DataFrames

@testset "End-to-end model comparison and stratified BSM" begin
    data = joinpath(@__DIR__, "data")
    baselines = joinpath(@__DIR__, "baselines")
    tr = read_tree(joinpath(data, "Psychotria_5.2.newick"))
    tips = read_lagrange_phylip(joinpath(data, "Psychotria_geog.data"))
    ss = StateSpace(tips.areanames; max_range_size = 4)
    refs = CSV.read(joinpath(baselines, "ml_psychotria.tsv"), DataFrame)
    specs = Dict("DEC" => :DEC, "DEC+J" => :DECJ, "DIVALIKE" => :DIVALIKE,
                 "DIVALIKE+J" => :DIVALIKEJ, "BAYAREALIKE" => :BAYAREALIKE,
                 "BAYAREALIKE+J" => :BAYAREALIKEJ)
    fits = [fit_model(tr, ss, tips, Model(specs[row.model]); maxeval = 2000,
                      calc_ancprobs = true) for row in eachrow(refs)]
    @test all(isapprox(f.loglik, r.lnL; atol = 1e-4) for (f, r) in zip(fits, eachrow(refs)))
    comparison = summary_table(fits; sort_by = :AICc)
    @test nrow(comparison) == 6
    @test issorted(comparison.AICc)
    @test comparison.model[1] == "DEC+J"
    @test sum(comparison.AICc_wt) ≈ 1
    r_aicc = Dict(row.model => aicc(row.lnL, row.numparams, ntips(tr)) for row in eachrow(refs))
    @test all(isapprox(row.AICc, r_aicc[row.model]; atol = 2e-4) for row in eachrow(comparison))
    @test all(all(isfinite, f.ancprobs.branch_top) for f in fits)
    @test_throws ArgumentError aicc(-10, 2, 3)

    d = joinpath(data, "psychotria_m0strat")
    strattr = read_tree(joinpath(d, "Psychotria_5.2.newick"))
    stratips = read_lagrange_phylip(joinpath(d, "Psychotria_geog.data"))
    stratss = StateSpace(stratips.areanames; max_range_size = 4)
    tc = tip_condlikes(strattr, stratips; max_range_size = 4)
    strat = Stratification(timesfn = joinpath(d, "timeperiods.txt"),
                           areas_allowed_fn = joinpath(d, "areas_allowed_all1s.txt"))
    model = Model(:DEC); model["d"] = 0.1; model["e"] = 0.05
    maps = run_bsm(strattr, stratss, model, strat, tc; seedval = 12344, n_maps = 2)
    @test [x.seedval for x in maps] == [12345, 12346]
    @test maps[1].node_states == [4, 4, 3, 3, 4, 4, 3, 2, 2, 4, 5, 3, 2, 2, 2, 2, 2, 2, 3,
                                  6, 2, 2, 12, 9, 9, 9, 4, 3, 4, 2, 2, 6, 3, 15, 2, 6, 6]
    @test maps[1].clado_event_txt[23] == "KOM->OM,K"
    r_events = [(10, 0.644559681043029, "e"), (18, 1.02398679224187, "e"),
                (34, 2.28493468679426, "d"), (34, 2.54318785724709, "d"),
                (23, 2.67103227926017, "d"), (32, 2.99438724185532, "d"),
                (23, 3.03492074785353, "d")]
    actual = sort([(n, e.abs_event_time, e.event_type) for n in 1:nnodes(strattr)
                   for e in maps[1].branch_events[n]], by = x -> x[2])
    @test [(x[1], x[3]) for x in actual] == [(x[1], x[3]) for x in r_events]
    @test [x[2] for x in actual] ≈ [x[2] for x in r_events] atol = 1e-7
    @test maps[2].node_states == stochastic_map_stratified(strattr, stratss, model, strat, tc; seedval = 12346).node_states
    @test_throws ArgumentError run_bsm(strattr, stratss, model, strat, tc; seedval = 1, n_maps = 0)
    tables = [bsm_tables(map, strattr) for map in maps]
    simulate_source_areas!(tables, stratss, RRng(2026))
    counts = count_ana_clado_events(tables, stratss)
    @test size(counts.summary_counts_BSMs) == (3, 12)
    @test counts.summary_counts_BSMs[3, 12] == sum(length(t.clado) + length(t.ana) for t in tables)

    # A fitted stratified model must also flow into the batch mapping API.
    fitted = fit_model(strattr, stratss, tc, Model(:DEC), strat;
                       algorithms = (:LN_BOBYQA,), maxeval = 80, calc_ancprobs = true)
    @test isfinite(fitted.loglik)
    @test fitted.ancprobs !== nothing
    fitted_maps = run_bsm(strattr, stratss, fitted.model, strat, tc; seedval = 2, n_maps = 1)
    @test length(bsm_tables(only(fitted_maps), strattr).clado) == ntips(strattr) - 1
end
